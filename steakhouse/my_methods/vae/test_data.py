import os
import pandas as pd
import numpy as np

def validate_dataset(filepath, name="Dataset", human_col='p0_action'):
    print(f"==========================================")
    print(f"🔍 INSPECTING: {name}")
    print(f"📁 File: {filepath}")
    print(f"==========================================")
    
    try:
        df = pd.read_pickle(filepath)
    except Exception as e:
        print(f"❌ Failed to load file: {e}")
        return

    # --- 1. COMPREHENSIVE SHAPE & KEY INTEGRITY CHECK ---
    print(f"📏 CHECKING DIMENSIONS:")
    keys_to_check = ['partial_obs_3d', 'full_obs_3d']
    
    for key in keys_to_check:
        if key in df.columns:
            # Check all shapes in the column
            shapes = [np.array(x).shape for x in df[key]]
            unique_shapes = set(shapes)
            
            if len(unique_shapes) == 1:
                actual_shape = list(unique_shapes)[0]
                print(f"  ✅ {key:15}: Consistent shape {actual_shape}")
                
                if 39 not in actual_shape:
                    print(f"     ⚠️  WARNING: 39 channels not found in {actual_shape}!")
            else:
                print(f"  ❌ {key:15}: MISMATCHED SHAPES! Found: {unique_shapes}")
        else:
            print(f"  ⚠️  {key:15}: Key missing from dataframe.")

    # Compare Partial vs Full if both exist
    if 'partial_obs_3d' in df.columns and 'full_obs_3d' in df.columns:
        p_shape = np.array(df['partial_obs_3d'].iloc[0]).shape
        f_shape = np.array(df['full_obs_3d'].iloc[0]).shape
        if p_shape != f_shape:
            print(f"  🚨 ALIGNMENT ERROR: Partial {p_shape} does not match Full {f_shape}!")

    # --- 2. LAYER VARIANCE SCANNER ---
    target_key = 'full_obs_3d' if 'full_obs_3d' in df.columns else 'partial_obs_3d'
    
    if target_key in df.columns:
        print(f"\n📡 SCANNING 39 LAYERS FOR VARIANCE (via {target_key}):")
        sample_size = min(1000, len(df))
        data_stack = np.stack(df[target_key].head(sample_size).values)
        
        # Determine Channel Axis
        shape = data_stack.shape # (Batch, D1, D2, D3)
        if shape[1] == 39:
            channel_axis = 1
            print("     (Format: Channel-First [C, H, W])")
        elif shape[3] == 39:
            channel_axis = 3
            print("     (Format: Channel-Last [H, W, C])")
        else:
            channel_axis = np.argmin(shape[1:]) + 1
            print(f"     (Assuming Channel axis is {channel_axis})")

        static_layers, dynamic_layers, timer_layers, all_zero_layers = [], [], [], []
        timer_layer_values = {} # NEW: Dictionary to store the non-0/1 values for timer layers
        num_channels = data_stack.shape[channel_axis]
        
        for c in range(num_channels):
            layer_data = data_stack[:, c, :, :] if channel_axis == 1 else data_stack[:, :, :, c]
            
            is_static = np.all(layer_data == layer_data[0])
            is_all_zero = np.max(layer_data) == 0
            
            if is_static:
                if is_all_zero: all_zero_layers.append(c)
                else: static_layers.append(c)
            else:
                dynamic_layers.append(c)
                unique_vals = np.unique(layer_data)
                if not np.all(np.isin(unique_vals, [0.0, 1.0, 0, 1])):
                    timer_layers.append(c)
                    
                    # NEW: Isolate and store the specific values that are not 0 or 1
                    non_binary_vals = unique_vals[~np.isin(unique_vals, [0.0, 1.0, 0, 1])]
                    timer_layer_values[c] = non_binary_vals.tolist()

        # --- 3. OUTPUT SUMMARY ---
        print(f"\n💀 DEAD WEIGHT:")
        print(f"  Empty (All Zeros)  [{len(all_zero_layers)}]: {all_zero_layers}")
        print(f"  Static (Layout)    [{len(static_layers)}]: {static_layers}")
        
        print(f"\n🔥 ACTIVE LAYERS:")
        print(f"  Dynamic (Changing)  [{len(dynamic_layers)}]: {dynamic_layers}")
        print(f"  Timer/Continuous    [{len(timer_layers)}]: {timer_layers}")
        
        # NEW: Print the exact values found in those timer layers
        if timer_layers:
            for layer_idx in timer_layers:
                print(f"       -> Layer {layer_idx} non-0/1 values: {timer_layer_values[layer_idx]}")

    # --- 4. ACTION CHECK ---
    if human_col in df.columns:
        actions = df[human_col].unique()
        print(f"\n🕹️  ACTIONS: {actions}")

    # --- 5. PARTIAL vs FULL EXACT MATCH CHECK ---
    # For every single row in the dataset, we compare whether the partial_obs_3d array
    # is byte-for-byte identical to the full_obs_3d array. We then count how many rows
    # are exact matches vs. how many differ.
    if 'partial_obs_3d' in df.columns and 'full_obs_3d' in df.columns:
        print(f"\n🔎 PARTIAL vs FULL EXACT MATCH CHECK:")
        
        total_rows = len(df)
        match_count = 0      # Counts rows where partial == full (array is identical)
        mismatch_count = 0   # Counts rows where partial != full (at least one value differs)
        mismatch_indices = []  # Stores the row numbers (indices) of mismatches for reference

        for idx in range(total_rows):
            # Convert each value to a numpy array so we can compare them element-by-element
            partial = np.array(df['partial_obs_3d'].iloc[idx])
            full    = np.array(df['full_obs_3d'].iloc[idx])

            # np.array_equal returns True only if both arrays have the same shape
            # AND every single element at every position is identical
            if np.array_equal(partial, full):
                match_count += 1
            else:
                mismatch_count += 1
                mismatch_indices.append(idx)  # Remember which row differed

        match_pct    = (match_count    / total_rows) * 100  # Percentage of matching rows
        mismatch_pct = (mismatch_count / total_rows) * 100  # Percentage of mismatching rows

        print(f"  Total Rows Checked : {total_rows}")
        print(f"  ✅ Exact Matches   : {match_count} ({match_pct:.1f}%)")
        print(f"  ❌ Mismatches      : {mismatch_count} ({mismatch_pct:.1f}%)")

        # If there are mismatches, print the first 10 row indices so you can investigate them
        if mismatch_indices:
            preview = mismatch_indices[:10]  # Show at most the first 10 mismatched row numbers
            print(f"  🔍 First mismatched row indices (up to 10): {preview}")
            if len(mismatch_indices) > 10:
                print(f"     ... and {len(mismatch_indices) - 10} more.")
    
    print("==========================================\n")

if __name__ == "__main__":
    base_path = "/Users/mishafu/Desktop/steakhouse/Steakhouse-AI/src/my_methods/data/fov_traj/"
    files = [        
        
        ("log_1edb8ed7.pkl", "SS-1"),
        
        # ("90_small_small1.pkl", "SS-1"),
        # ("90_small_small2.pkl", "SS-2"),
        # #("90_small_small3.pkl", "SS-3"),
        # #("90_small_small4.pkl", "SS-4"),
        # ("90_small_small5.pkl", "SS-5"),
        # ("full_90_2_2-4.pkl", "SS2-4"),
        # ("full_179_2_2-5.pkl", "SS2-5"),
    ]
    
    for filename, nickname in files:
        full_path = os.path.join(base_path, filename)
        if os.path.exists(full_path):
            validate_dataset(full_path, name=nickname)
        else:
            print(f"❌ Missing file: {filename}")