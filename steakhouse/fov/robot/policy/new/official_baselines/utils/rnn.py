
"""
RNN layer
ref: https://github.com/sjtu-marl/ZSC-Eval/tree/master
     zsceval/algorithms/utils/rnn.py

x  = "what's going on right now, memory included"   -> ACTLayer picks an action from this
h  = "the memory itself, to reuse next tick"        -> goes back in the buffer

===========================================================================
NOTATIONS -- every letter used in every comment in this file
===========================================================================
A tick (or "frame") = one timestep of the game. One state, one call to
build_full_state, one (23, W, H) observation. I say tick from here on.

N = how many kitchens you're running at the same time.
    To collect data faster you don't play one game, you play N independent
    copies side by side. Kitchen 5 knows nothing about kitchen 6.
    NEVER hardcoded in this file. Always read at runtime as hxs.size(0),
    because we genuinely do not know it here.
    (ZSC-Eval default n_rollout_threads = 32, just for a sense of scale.)

T = how many ticks in a row you look at. AKA CONTINUOUS TICKS OR NUMBER OF STEPS
    ONLY exists while training. Computed as x.size(0) // N.
    (ZSC-Eval default data_chunk_length = 10.)

L = how many GRU layers are stacked = recurrent_N_layers. Default 1.

H = hidden size = how many numbers are in one memory vector. 64 here.

F = input feature size = whatever CNNBase.output_size is. 64 here.

===========================================================================
INPUT -- and why the shapes differ between the two branches
===========================================================================
forward() is only ever called in two situations, and they hand it
different shapes. That is the ENTIRE reason for the if/else.

  PLAYING (collecting data):
      all N kitchens sit at the same tick and each needs one action now.
      there is only ONE tick, so there is no T at all.

  TRAINING (learning from data you already collected):
      you replay a chunk of T consecutive ticks from all N kitchens at
      once -- that's T*N observations. The buffer stores them squashed
      into one flat list of rows, because the CNN and the Linear layers
      don't care about time, they just process rows. Only the GRU cares
      about time, so only the GRU un-squashes.

x      the CNN output.
       PLAYING : (N,   F)     one tick from each kitchen
       TRAINING: (T*N, F)     T ticks from each kitchen, squashed flat

hxs    the memory ("hidden state"). ALWAYS (N, L, H) in BOTH cases.
       One memory per kitchen. There is no tick axis, because memory is a
       running summary of the past, not a list of the past.
       ==> this is exactly what the if/else tests. hxs.size(0) is always N:
             x.size(0) == hxs.size(0)  means  N   == N   -> PLAYING
             x.size(0) != hxs.size(0)  means  T*N != N   -> TRAINING

masks  a 0/1 flag per kitchen: "is this the same episode as last tick?"
         1 = same episode   -> keep the memory
         0 = just restarted -> wipe the memory
       PLAYING : (N,   1)
       TRAINING: (T*N, 1)
       Skip this and a kitchen that just finished an episode carries its
       old memory into the brand new one. Nothing crashes, nothing warns,
       the agent just learns from contaminated memory forever.

===========================================================================
OUTPUT
===========================================================================
x      same number of rows it came in with, but H wide:
       PLAYING (N, H)  /  TRAINING (T*N, H)
hxs    (N, L, H) -- flipped back into the format the buffer handed us

===========================================================================
THE SHAPE OPERATIONS USED BELOW
===========================================================================
A tensor is a flat run of numbers plus a shape telling you how to group
them. The six numbers  a b c d e f  can be worn as:

    shape (6,)       a b c d e f
    shape (2, 3)     [[a b c] [d e f]]
    shape (3, 2)     [[a b] [c d] [e f]]
    shape (1, 2, 3)  [[[a b c] [d e f]]]

Same six numbers, same order. Only the grouping differs.

.view(...)      Regroup. The ORDER of the numbers never changes, nothing
                moves, it costs nothing. A -1 means "figure this axis out
                from the total count":
                    32 numbers .view(-1, 1, 1)  ->  (32, 1, 1) instead of (1, 32,  1)

.reshape(...)   Exactly like .view, with one difference: .view REFUSES to
                run when the numbers aren't laid out contiguously in
                memory (which is what a .transpose leaves behind), while
                .reshape quietly makes a copy instead of erroring.
                So: reshape = view that never fails.

.transpose(a,b) SWAPS two axes. Unlike view, this one really does change
                the order the numbers come out in:
                    (2,3)  [[a b c] [d e f]]
                      -> transpose(0,1) ->
                    (3,2)  [[a d] [b e] [c f]]
                view CANNOT do this. transpose is the only tool that can.

.unsqueeze(i)   Insert a new axis of size 1 at position i. (N,F) -> (1,N,F)
.squeeze(i)     Delete an axis of size 1 at position i.    (1,N,F) -> (N,F)

.contiguous()   After a transpose the numbers are in the right logical
                order but scrambled in physical memory. This re-lays them
                out. nn.GRU wants its memory argument laid out this way.

===========================================================================
WHY  masks.view(-1, 1, 1)  AND NOT JUST  hxs * masks
===========================================================================
When you multiply two tensors of different shapes, torch lines them up
from the RIGHT and pads the shorter one with 1s on the LEFT. Then every
axis must either match, or one side must be 1 (which gets stretched).

Plain hxs * masks, using N=32, L=1, H=64 to make it concrete:

    hxs    (N, L, H)  =  (32,  1, 64)
    masks  (N, 1)     =  (32,  1)     -> padded to (1, 32,  1)

    axis 0:   32 vs  1   ->  stretches to 32
    axis 1:    1 vs 32   ->  stretches to 32    <-- THIS IS THE BUG
    axis 2:   64 vs  1   ->  stretches to 64
                             result (32, 32, 64)

That result is 32x too big and it has multiplied kitchen i's flag against
kitchen j's memory. Torch does NOT raise an error. Silently, plausibly
wrong -- the worst kind.

With the view:

    hxs                   (32, 1, 64)
    masks.view(-1, 1, 1)  (32, 1,  1)

    axis 0:   32 vs 32   ->  match
    axis 1:    1 vs  1   ->  match
    axis 2:   64 vs  1   ->  stretches to 64
                             result (32, 1, 64)     CORRECT

Each kitchen's own flag now hits its own memory and nothing else.
.view(-1,1,1) also works whether masks arrives as (N,) or (N,1), because
both hold exactly N numbers and the -1 absorbs whichever it is.

===========================================================================
WHY THE MEMORY KEEPS FLIPPING  (N, L, H) <-> (L, N, H)
===========================================================================
Two pieces of code want the memory laid out differently and neither will
budge, so you flip on the way in and flip back on the way out:

    the rollout buffer stores  (N, L, H)   kitchen first, so that hxs[k]
                                           is kitchen k's memory
    nn.GRU demands             (L, N, H)   layer first. This is simply
                                           PyTorch's documented choice.

transpose(0,1) swaps those first two axes. view cannot do this job,
because the numbers genuinely have to come out in a different order.

===========================================================================
WHERE THIS SITS IN THE STACK
===========================================================================
──────────── CNNBase  (cnn.py)  ← DONE, you built this
Conv2d(23, 32, k=3,s=1,p=1) → ReLU
Conv2d(32, 64, k=3,s=1,p=1) → ReLU
Conv2d(64, 32, k=3,s=1,p=1) → ReLU
Flatten
Linear(32*W*H, 64) → ReLU → LayerNorm(64)
Linear(64, output_size) → ReLU → LayerNorm(output_size)
                                        emits (N, 64)

──────────── RNNLayer  (rnn.py)  ← THIS FILE
GRU(64, 64, num_layers=1)
LayerNorm(64)
                                        emits (N, 64) + new hidden state

──────────── MLPLayer  (mlp.py)  ← SKIP THIS
(not used: layer_after_N defaults to 0)

──────────── ACTLayer  (act.py)  ← after RNN
Linear(64, 6) → Categorical
                                        emits action, log_prob
"""

import torch
import torch.nn as nn

class RNNLayer(nn.Module):
    def __init__(self, inputs_dim=64, hidden_dim=64, recurrent_N_layers=1):

        super().__init__()

        #L. stored because forward() must not hardcode it either.
        self.recurrent_N_layers = recurrent_N_layers

        #nn.GRU(input_size, hidden_size, num_layers)  ==  nn.GRU(F, H, L)
        self.rnn = nn.GRU(inputs_dim, hidden_dim, num_layers=recurrent_N_layers)
        #normalizes the H numbers inside each output vector. no shape change.
        self.norm = nn.LayerNorm(hidden_dim)


        #also use orthogonal. GRU gives named params weight and biases.
        #a GRU is NOT one matrix -- per layer it holds FOUR tensors:
        #    weight_ih_l0  (3H, F)  2-D grid, turns the input into gates
        #    weight_hh_l0  (3H, H)  2-D grid, turns the memory into gates
        #    bias_ih_l0    (3H,)    1-D list
        #    bias_hh_l0    (3H,)    1-D list
        #the 3 is because a GRU computes three things per tick: a reset gate,
        #an update gate, and a candidate memory.
        #this HAS to be a loop and not one init call, because orthogonal_ only
        #accepts 2-D or bigger -- handing it a 1-D bias list throws.
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)          # flat list -> just zeros
            elif "weight" in name:
                nn.init.orthogonal_(param)           # 2-D grid -> orthogonal


    def forward(self, x, hxs, masks):

        #am I in test time or training time?
        #Playing: one frame from each of N kitchens.
        #Training: you replay T frames from each of N kitchens at once

        if x.size(0) == hxs.size(0): #playing, one frame per env.  x (N, feat), hxs (N, layers, H)
            #hxs   has 3 axes:  (N, L, H)  =  (32, 1, 64)
            #                        ^   ^   ^
            #                        |   |   └─ 64 memory numbers → spread → put 1
            #                        |   └───── 1 layer           → spread → put 1
            #                        └───────── 32 kitchens       → real   → put 32

            # so masks must be:              (32, 1,  1)
            #The -1 means "count the numbers yourself" — there are 32, so it eg. 32. That's why the code never has to know it's like 32 per say.
            #masks arrives as (32, 1) — only 2 axes, not 3. Torch pads missing axes on the left, so it becomes (1, 32, 1) so 32 is in the number of layer dimension
            #FIX: was "h = h * ..." but h does not exist yet -- this line is
            #where h is BORN. it has to read from hxs, the argument.
            h = hxs * masks.view(-1, 1, 1)        # 0 erases the memory, 1 leaves it alone

            #GRU format is always LxNxH
            #so like
            #out, h_n = self.rnn(x, h)
            # x    (T, N, F)  ->  out  (T, N, H)     one output per tick
            # h    (L, N, H)  ->  h_n  (L, N, H)     memory after the last tick
            #but for the PPO rollout buffer that store everything they always go env first (N first) so we gotta do this
            h = h.transpose(0, 1).contiguous()  #(32,1,64) -> (1,32,64) for the GRU

            #expects (how many frames per kitchen, how many different kitchen, how many features)
            #or (layers, kitchens, features)
            #but instead we have N, 64
            x = x.unsqueeze(0)     # (N, 64)  ->  (1, N, 64)   "one frame"

            #FIX: was "x, hxs = self.rnn(x, hxs)". two problems with that:
            #  1. it fed the GRU the RAW hxs -- throwing away both the mask
            #     wipe and the transpose you just did on the line above
            #  2. it overwrote hxs with the GRU's output, so the transpose
            #     back on the last line had nothing left to work with
            #feed h, catch h. hxs stays untouched until the very end.
            x, h = self.rnn(x, h)  # x -> (1, N, 64),  h -> (L, N, 64)

            x = x.squeeze(0)       # (1, N, 64) -> (N, 64)     take it back off

            #ok now we transform back...
            hxs = h.transpose(0, 1)   # (L, N, H) -> (N, L, H)  buffer layout

        else: #training

            # ------------- TRAINING: T ticks, N kitchens -------------
            #             # x  = (T*N, 64)    T ticks per kitchen, squashed into a flat list
            #             # hxs = (N, 1, 64) still just one memory per kitchen
            #             # masks (T*N, 1)
            
            
            #number of kitchens
            N = hxs.size(0)
            #number of ticks per kitchen
            #FIX: was "T = x.size(0)/N". In python 3 the / operator always
            #gives a FLOAT (320/32 -> 10.0), and range() refuses a float.
            #int(...) makes it a whole number.
            T = int(x.size(0) / N)

            x = x.view(T, N, x.size(1))     # (T*N, F) -> (T, N, F)
            #FIX: was "mask = mask.view(T, N)". The argument is called masks,
            #with an s. "mask" was never defined -> NameError.
            masks = masks.view(T, N)        # (T*N, 1) -> (T, N)

            #FIX: was "h = h.transpose(...)". Same bug as the playing branch --
            #h does not exist yet in this branch either. Read from hxs.
            h = hxs.transpose(0, 1).contiguous()   # (N, L, H) -> (L, N, H)

            #collects the GRU output from every tick, so we can hand the actor
            #a result for all T ticks and not just the last one
            outputs = []

            #So for t in range(T) walks time, and every single iteration handles all N kitchens simultaneously. There is no for n in range(N) because there never needs to be.
            #but mask[t] is per N
            # t = 0:  masks[0] = [1, 1, 0, 1]    wipe kitchen 2's memory
            #         x[0]     = (4, 64)          all 4 kitchens' features at tick 0
            #         -> one GRU call -> h updated for all 4 kitchens

            # t = 1:  masks[1] = [1, 1, 1, 1]    wipe nobody
            #         x[1]     = (4, 64)          all 4 kitchens' features at tick 1
            #         -> one GRU call -> h updated for all 4 kitchens

            # t = 2:  masks[2] = [1, 0, 1, 1]    wipe kitchen 1's memory
            #         x[2]     = (4, 64)          all 4 kitchens' features at tick 2
            #         -> one GRU call -> h updated for all 4 kitchens
            
            for t in range(T):
                #masks[t] is (N,) -- every kitchen's flag AT TICK t.
                #view(1,-1,1) -> (1,N,1) because h is (L,N,H) and N sits in
                #the MIDDLE here. (in the playing branch hxs was (N,L,H) and N
                #was first, which is why that one used view(-1,1,1) instead.)
                h = h * masks[t].view(1, -1, 1)

                #x[t] is (N, F) -- one tick across every kitchen at once.
                #unsqueeze(0) -> (1, N, F), the "exactly one tick" announcement.
                out, h = self.rnn(x[t].unsqueeze(0), h)   # out (1, N, H)

                #FIX: this line was missing. without it outputs stays empty and
                #every tick's result is thrown away the moment the next loop
                #starts. h survives (it is reassigned), but the OUTPUT does not.
                outputs.append(out)
                #h carries into the next iteration -- that IS the memory working

            #FIX: everything below was missing. after the loop, x is still the
            #(T, N, F) INPUT -- the GRU results are sitting in the outputs list
            #and were never put back into x.
            x = torch.cat(outputs, dim=0)   # T tensors of (1,N,H) -> (T, N, H)
            x = x.reshape(T * N, -1)        # (T, N, H) -> (T*N, H) flat again,
                                            # matching the shape it arrived in
            hxs = h.transpose(0, 1)         # (L, N, H) -> (N, L, H) buffer layout

        x = self.norm(x)

        return x, hxs
