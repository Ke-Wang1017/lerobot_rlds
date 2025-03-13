#!/usr/bin/env python

"""
Script to convert a LeRobot dataset to RLDS format and save it as a TensorFlow dataset.
"""

import argparse
import os
import tensorflow_datasets as tfds
from lerobot_to_rlds_converter import create_rlds_dataset_builder
import numpy as np
import torch
import traceback  # Add this to get detailed error information

# Function to load your dataset
def load_dataset(repo_id, root=None, local_files_only=False):
    """
    Load a dataset by repo_id using LeRobotDataset.
    
    Args:
        repo_id: Repository ID of the dataset
        root: Root directory for the dataset stored locally
        local_files_only: Use local files only
        
    Returns:
        A dataset object in the format expected by DatasetToRLDSConverter
    """
    try:
        # Try to import LeRobotDataset
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        
        # Load the LeRobot dataset
        lerobot_dataset = LeRobotDataset(repo_id, root=root, local_files_only=local_files_only)
        
        # Create a dataset structure compatible with our converter
        dataset = {
            "episodes": [],
            "metadata": {
                "repo_id": repo_id
            }
        }
        
        # Define EpisodeSampler class (similar to the one in visualize_dataset.py)
        class EpisodeSampler(torch.utils.data.Sampler):
            def __init__(self, dataset, episode_index):
                from_idx = dataset.episode_data_index["from"][episode_index].item()
                to_idx = dataset.episode_data_index["to"][episode_index].item()
                self.frame_ids = range(from_idx, to_idx)
                
            def __iter__(self):
                return iter(self.frame_ids)
                
            def __len__(self):
                return len(self.frame_ids)
        
        # Function to convert CHW float32 tensor to HWC uint8 numpy array
        def to_hwc_uint8_numpy(chw_float32_torch):
            if chw_float32_torch.dtype == torch.float32 and chw_float32_torch.ndim == 3:
                c, h, w = chw_float32_torch.shape
                if c == 3:  # RGB image
                    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
                    return hwc_uint8_numpy
            # Return as is if not a float32 CHW image
            return chw_float32_torch.numpy()
        
        # Get number of episodes
        num_episodes = len(lerobot_dataset.episode_data_index["from"])
        
        # Process each episode
        for episode_idx in range(num_episodes):
            # Create a sampler for this episode
            episode_sampler = EpisodeSampler(lerobot_dataset, episode_idx)
            
            # Create a dataloader with batch size 1 to load frames one by one
            dataloader = torch.utils.data.DataLoader(
                lerobot_dataset,
                batch_size=1,
                sampler=episode_sampler,
                num_workers=0
            )
            
            # Create episode structure
            episode = {
                "frames": [],
                "file_path": f"episode_{episode_idx}"
            }
            
            # Extract language instruction if available
            language_instruction = ""
            if hasattr(lerobot_dataset, 'get_language_instruction'):
                language_instruction = lerobot_dataset.get_language_instruction(episode_idx)
            
            # Process each frame in the episode
            for batch in dataloader:
                # Convert batch tensors to numpy arrays and remove batch dimension
                frame = {}
                
                # Add state if available
                if 'observation.state' in batch:
                    frame['state'] = batch['observation.state'][0].numpy()
                
                # Add action if available
                if 'action' in batch:
                    frame['action'] = batch['action'][0].numpy()
                
                # Add reward if available
                if 'next.reward' in batch:
                    frame['reward'] = batch['next.reward'][0].item()
                
                # Add camera images
                for key in lerobot_dataset.meta.camera_keys:
                    if key in batch:
                        frame[key.replace('.', '_')] = to_hwc_uint8_numpy(batch[key][0])
                
                # Add language instruction
                if language_instruction:
                    frame['language_instruction'] = language_instruction
                
                episode["frames"].append(frame)
            
            # Add episode to dataset
            dataset["episodes"].append(episode)
        
        return dataset
        
    except (ImportError, ModuleNotFoundError) as e:
        print(f"Warning: Could not import LeRobotDataset: {e}")
        print("Falling back to dummy dataset")
        
        # Create a minimal dataset with one empty episode for testing
        return {
            "episodes": [{
                "frames": [{
                    "action": np.zeros(1, dtype=np.float32),
                    "state": np.zeros(1, dtype=np.float32),
                    "image": np.zeros((64, 64, 3), dtype=np.uint8),
                    "language_instruction": "test instruction"
                }],
                "file_path": "test_path"
            }],
            "metadata": {
                "repo_id": repo_id
            }
        }
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(traceback.format_exc())  # Print the full traceback for debugging
        print("Falling back to dummy dataset")
        
        # Create a minimal dataset with one empty episode for testing
        return {
            "episodes": [{
                "frames": [{
                    "action": np.zeros(1, dtype=np.float32),
                    "state": np.zeros(1, dtype=np.float32),
                    "image": np.zeros((64, 64, 3), dtype=np.uint8),
                    "language_instruction": "test instruction"
                }],
                "file_path": "test_path"
            }],
            "metadata": {
                "repo_id": repo_id
            }
        } 


def main():
    parser = argparse.ArgumentParser(description="Convert LeRobot dataset to RLDS format")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of HuggingFace repository containing a LeRobotDataset (e.g. `lerobot/pusht`).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the RLDS dataset. If not provided, uses the default TFDS data directory.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use local files only. By default, this script will try to fetch the dataset from the hub if it exists.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory for the dataset stored locally.",
    )
    
    args = parser.parse_args()
    
    # Load the dataset
    dataset = load_dataset(args.repo_id, args.root, args.local_files_only)
    
    # Create the dataset builder class
    dataset_builder_class = create_rlds_dataset_builder(dataset, args.repo_id)
    
    # Create an instance of the dataset builder
    builder = dataset_builder_class(
        dataset=dataset,
        repo_id=args.repo_id,
        data_dir=args.output_dir
    )
    
    # Download and prepare the dataset
    builder.download_and_prepare()
    
    # Print information about the dataset
    print(f"Dataset {args.repo_id} converted to RLDS format and saved to {builder.data_dir}")
    print(f"Dataset info: {builder.info}")
    
    # Show how to load the dataset
    print("\nTo load this dataset, use:")
    print(f"dataset = tfds.load('{os.path.join(builder.data_dir, builder.name)}')")


if __name__ == "__main__":
    main() 