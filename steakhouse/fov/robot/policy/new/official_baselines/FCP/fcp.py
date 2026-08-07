"""
Third after e3t.

FCP is the natural second choice but you have a concrete blocker. FCP's stage 1 needs a population of checkpoints at different skill levels — early, mid, late — from SP runs. Your --save_interval 25 overwrote the same filename every time, so you only have the final weights. You'd have to re-run SP with per-episode checkpoint filenames before FCP is even possible. Worth knowing before you plan around it.

"""