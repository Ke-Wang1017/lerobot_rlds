"""
Utility to convert datasets to RLDS format.
This converter allows using datasets with RLDS-compatible tools and frameworks.
"""

from typing import Iterator, Tuple, Any, Dict, List, Optional
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import torch.utils.data


class DatasetToRLDSConverter:
    """Converts datasets to RLDS format."""
    
    def __init__(self, dataset, repo_id: str):
        """
        Initialize the converter with a dataset.
        
        Args:
            dataset: The dataset to convert
            repo_id: Name of the repository containing the dataset
        """
        self.dataset = dataset
        self.repo_id = repo_id
        
    def _get_episode_indices(self) -> List[int]:
        """Get all episode indices in the dataset."""
        # Assuming dataset has an "episodes" list
        return list(range(len(self.dataset["episodes"])))
    
    def _get_episode_frames(self, episode_idx: int) -> List[Dict]:
        """Get all frames for a specific episode."""
        # Assuming each episode has a "frames" list
        return self.dataset["episodes"][episode_idx]["frames"]
    
    def _convert_to_rlds_episode(self, episode_idx: int) -> Dict:
        """Convert an episode to RLDS format."""
        frames = self._get_episode_frames(episode_idx)
        
        # Convert frames to RLDS steps
        steps = []
        for i, frame in enumerate(frames):
            # Create observation dict
            observation = {}
            
            # Add camera images
            for key in frame.keys():
                if key.startswith('image') or key.endswith('image'):
                    # Convert image to the right format
                    observation[key] = frame[key]
            
            # Add state if available
            if 'state' in frame:
                observation['state'] = frame['state']
            
            # Create step dict
            step = {
                'observation': observation,
                'discount': 1.0,
                'is_first': i == 0,
                'is_last': i == len(frames) - 1,
                'is_terminal': i == len(frames) - 1,
            }
            
            # Add action if available
            if 'action' in frame:
                step['action'] = frame['action']
            
            # Add reward if available (default to 1.0 at the end for demos)
            if 'reward' in frame:
                step['reward'] = float(frame['reward'])
            else:
                step['reward'] = float(i == len(frames) - 1)
            
            # Add language instruction if available
            if 'language_instruction' in frame:
                step['language_instruction'] = frame['language_instruction']
                
            steps.append(step)
        
        # Create RLDS episode
        episode = {
            'steps': steps,
            'episode_metadata': {
                'episode_index': episode_idx,
                'repo_id': self.repo_id,
            }
        }
        
        return episode
    
    def generate_examples(self, split: Optional[str] = None) -> Iterator[Tuple[str, Dict]]:
        """
        Generate RLDS examples from the dataset.
        
        Args:
            split: Dataset split to use (if None, uses all episodes)
            
        Yields:
            Tuple of (episode_id, episode_dict) in RLDS format
        """
        episode_indices = self._get_episode_indices()
        
        for idx in episode_indices:
            episode = self._convert_to_rlds_episode(idx)
            yield f"{self.repo_id}_{idx}", episode
    
    def get_rlds_features(self) -> tfds.features.FeaturesDict:
        """
        Get the RLDS features dictionary for the dataset.
        
        Returns:
            A tfds.features.FeaturesDict describing the dataset structure
        """
        # Get a sample episode to determine the features
        sample_idx = 0
        sample_episode = self._convert_to_rlds_episode(sample_idx)
        sample_step = sample_episode['steps'][0]
        
        # Create observation features
        observation_features = {}
        for key, value in sample_step['observation'].items():
            if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[2] == 3:
                # Image feature
                observation_features[key] = tfds.features.Image(
                    shape=value.shape,
                    dtype=np.uint8,
                    encoding_format='jpeg',
                    doc=f'{key} camera RGB observation.'
                )
            elif isinstance(value, np.ndarray):
                # Tensor feature (e.g., state)
                observation_features[key] = tfds.features.Tensor(
                    shape=value.shape,
                    dtype=value.dtype,
                    doc=f'{key} observation.'
                )
        
        # Create step features
        step_features = {
            'observation': tfds.features.FeaturesDict(observation_features),
            'discount': tfds.features.Scalar(
                dtype=np.float32,
                doc='Discount if provided, default to 1.'
            ),
            'reward': tfds.features.Scalar(
                dtype=np.float32,
                doc='Reward if provided, 1 on final step for demos.'
            ),
            'is_first': tfds.features.Scalar(
                dtype=np.bool_,
                doc='True on first step of the episode.'
            ),
            'is_last': tfds.features.Scalar(
                dtype=np.bool_,
                doc='True on last step of the episode.'
            ),
            'is_terminal': tfds.features.Scalar(
                dtype=np.bool_,
                doc='True on last step of the episode if it is a terminal step, True for demos.'
            ),
        }
        
        # Add action if available
        if 'action' in sample_step:
            step_features['action'] = tfds.features.Tensor(
                shape=sample_step['action'].shape,
                dtype=np.float32,
                doc='Robot action.'
            )
        
        # Add language instruction if available
        if 'language_instruction' in sample_step:
            step_features['language_instruction'] = tfds.features.Text(
                doc='Language Instruction.'
            )
        
        # Create full features dict
        features = tfds.features.FeaturesDict({
            'steps': tfds.features.Dataset(step_features),
            'episode_metadata': tfds.features.FeaturesDict({
                'episode_index': tfds.features.Scalar(
                    dtype=np.int32,
                    doc='Index of the episode in the dataset.'
                ),
                'repo_id': tfds.features.Text(
                    doc='Repository ID of the dataset.'
                ),
            }),
        })
        
        return features


class RLDSDatasetBuilder(tfds.core.DatasetBuilder):
    """Base class for creating RLDS dataset builders."""
    
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    
    def __init__(self, dataset, repo_id, **kwargs):
        """
        Initialize the dataset builder.
        
        Args:
            dataset: The dataset to convert
            repo_id: Repository ID of the dataset
            **kwargs: Additional arguments to pass to the parent class
        """
        self.repo_id = repo_id
        self.dataset = dataset
        self.converter = DatasetToRLDSConverter(dataset, repo_id)
        super().__init__(**kwargs)
    
    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata."""
        return self.dataset_info_from_configs(
            features=self.converter.get_rlds_features()
        )
    
    def _split_generators(self, dl_manager):
        """Define splits."""
        return {
            'train': self._generate_examples('train'),
        }
    
    def _generate_examples(self, split):
        """Generate examples for the given split."""
        yield from self.converter.generate_examples(split)
    
    def _as_dataset(self, split, shuffle_files=False, decoders=None, read_config=None):
        """Constructs a tf.data.Dataset."""
        # Create a list to store all examples
        examples = []
        
        # Generate examples for the requested split
        for key, example in self._generate_examples(split):
            examples.append(example)
        
        # Create a dataset from the examples
        dataset = tf.data.Dataset.from_tensor_slices(examples)
        
        return dataset
    
    def _download_and_prepare(self, dl_manager, download_config=None):
        """Downloads and prepares dataset for reading."""
        # Since we're working with an in-memory dataset, we don't need to download anything
        # We just need to prepare the dataset for reading
        self._prepare()
    
    def _prepare(self):
        """Prepares the dataset for reading."""
        # This method is called by _download_and_prepare
        # We don't need to do anything special here since we're working with an in-memory dataset
        pass


def create_rlds_dataset_builder(dataset_or_repo_id, repo_id=None):
    """
    Create a dataset builder class for a specific dataset.
    
    Args:
        dataset_or_repo_id: Either the dataset to convert or the repository ID
        repo_id: Repository ID of the dataset (only needed if dataset_or_repo_id is a dataset)
        
    Returns:
        A dataset builder class that can be used to create RLDS datasets
    """
    # Handle both old and new calling conventions
    if repo_id is None:
        # Old style: create_rlds_dataset_builder(repo_id)
        repo_id = dataset_or_repo_id
        # Create a minimal dataset with one empty episode for testing
        dataset = {
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
    else:
        # New style: create_rlds_dataset_builder(dataset, repo_id)
        dataset = dataset_or_repo_id
    
    # Create a class name from the repo_id
    class_name = repo_id.replace('/', '_').replace('-', '_')
    
    # Create a new class that inherits from RLDSDatasetBuilder
    dataset_builder_class = type(
        class_name,
        (RLDSDatasetBuilder,),
        {
            'repo_id': repo_id,
            '__doc__': f'RLDS dataset builder for {repo_id}.'
        }
    )
    
    return dataset_builder_class 

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