#!/usr/bin/env python3
"""
Process all MOT17 and MOT20 videos with accumulated motion prediction.
Organizes outputs by video name and saves comprehensive evaluation results.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime
import shutil

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from accumulated_motion_predictor import AccumulatedMotionPredictor

class DatasetProcessor:
    """Process all MOT17 and MOT20 videos with comprehensive evaluation."""
    
    def __init__(self, base_output_dir="dataset_evaluation_results"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_output_dir / f"run_{timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.videos_dir = self.run_dir / "videos"
        self.results_dir = self.run_dir / "results"
        self.summary_dir = self.run_dir / "summary"
        
        for dir_path in [self.videos_dir, self.results_dir, self.summary_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Results storage
        self.all_results = []
        self.dataset_summary = {}
        
        print(f"🗂️  Output directory: {self.run_dir}")
        print(f"📹 Videos will be saved to: {self.videos_dir}")
        print(f"📊 Results will be saved to: {self.results_dir}")
        print(f"📈 Summary will be saved to: {self.summary_dir}")
    
    def get_available_sequences(self):
        """Get all available MOT17 and MOT20 sequences."""
        predictor = AccumulatedMotionPredictor(verbose=False)
        
        # Get all available sequences - check MOT17 and MOT20
        success_mot17, sequences_mot17 = predictor.load_sequence()
        sequences = sequences_mot17 if success_mot17 else []
        
        # Try to get MOT20 sequences by modifying the factory
        try:
            sequences_mot20 = predictor.factory.list_sequences(['MOT20'], ['960x960'])
            sequences.extend(sequences_mot20)
        except:
            pass  # MOT20 might not be available
        
        if not sequences:
            print("❌ No sequences found!")
            return []
        
        # Filter for MOT17 and MOT20
        target_datasets = ['MOT17', 'MOT20']
        filtered_sequences = []
        
        for seq in sequences:
            for dataset in target_datasets:
                if dataset in seq:
                    filtered_sequences.append(seq)
                    break
        
        print(f"🎯 Found {len(filtered_sequences)} sequences:")
        for i, seq in enumerate(filtered_sequences, 1):
            print(f"   {i:2d}. {seq}")
        
        return filtered_sequences
    
    def extract_video_info(self, sequence_name):
        """Extract video information from sequence name."""
        # Example: MOT17-04-SDP_960x960_gop50_500frames
        parts = sequence_name.split('_')
        if len(parts) >= 1:
            video_base = parts[0]  # MOT17-04-SDP
            dataset = video_base.split('-')[0]  # MOT17
            video_id = '-'.join(video_base.split('-')[1:])  # 04-SDP
            return {
                'dataset': dataset,
                'video_id': video_id,
                'video_name': video_base,
                'full_sequence': sequence_name
            }
        return None
    
    def process_single_video(self, sequence_name, num_gops=3):
        """Process a single video sequence."""
        video_info = self.extract_video_info(sequence_name)
        if not video_info:
            print(f"❌ Could not parse sequence name: {sequence_name}")
            return None
        
        print(f"\n{'='*80}")
        print(f"🎬 Processing: {video_info['video_name']}")
        print(f"   Dataset: {video_info['dataset']}")
        print(f"   Video ID: {video_info['video_id']}")
        print(f"   Full sequence: {sequence_name}")
        print(f"{'='*80}")
        
        # Create video-specific directory
        video_dir = self.videos_dir / video_info['video_name']
        video_dir.mkdir(exist_ok=True)
        
        # Initialize predictor for this sequence
        predictor = AccumulatedMotionPredictor(verbose=True)
        
        # Load specific sequence
        success, _ = predictor.load_sequence(sequence_name)
        if not success:
            print(f"❌ Failed to load sequence: {sequence_name}")
            return None
        
        # Process GOPs and get results
        try:
            # Set output prefix to save videos in video-specific directory
            original_cwd = os.getcwd()
            os.chdir(video_dir)
            
            gop_results = predictor.process_gops(num_gops=num_gops, sequence_prefix=video_info['video_name'])
            
            # Return to original directory
            os.chdir(original_cwd)
            
            if not gop_results:
                print(f"❌ No results obtained for {video_info['video_name']}")
                return None
            
            # Calculate video-level metrics
            video_metrics = self.calculate_video_metrics(gop_results, video_info)
            
            # Save individual video results
            self.save_video_results(video_metrics, video_info)
            
            print(f"✅ Completed processing: {video_info['video_name']}")
            print(f"   📹 Videos saved to: {video_dir}")
            print(f"   📊 Mean mAP: {video_metrics['overall_mean_mAP']:.3f}")
            
            return video_metrics
            
        except Exception as e:
            print(f"❌ Error processing {video_info['video_name']}: {str(e)}")
            return None
    
    def calculate_video_metrics(self, gop_results, video_info):
        """Calculate comprehensive metrics for a video."""
        all_gop_maps = []
        all_gop_ap50s = []
        all_gop_ap75s = []
        total_objects = 0
        all_object_results = []
        
        # Collect data from all GOPs
        for gop_idx, gop_data in gop_results.items():
            if 'mean_metrics' in gop_data:
                metrics = gop_data['mean_metrics']
                all_gop_maps.append(metrics['mean_mAP'])
                all_gop_ap50s.append(metrics['mean_AP50'])
                all_gop_ap75s.append(metrics['mean_AP75'])
                total_objects += metrics['num_objects']
                
                # Collect individual object results
                if 'individual_results' in gop_data:
                    for obj_id, obj_result in gop_data['individual_results'].items():
                        obj_data = {
                            'gop': gop_idx,
                            'object_id': obj_id,
                            'mAP': obj_result['mAP'],
                            'AP50': obj_result['AP_per_threshold'][0.5],
                            'AP75': obj_result['AP_per_threshold'][0.75],
                            'predictions': obj_result['total_predictions'],
                            'ground_truths': obj_result['total_ground_truths']
                        }
                        all_object_results.append(obj_data)
        
        # Calculate overall video metrics
        video_metrics = {
            'video_info': video_info,
            'timestamp': datetime.now().isoformat(),
            'num_gops_processed': len(all_gop_maps),
            'total_objects_tracked': total_objects,
            'gop_metrics': {
                'individual_gop_mAPs': all_gop_maps,
                'individual_gop_AP50s': all_gop_ap50s,
                'individual_gop_AP75s': all_gop_ap75s
            },
            'overall_mean_mAP': np.mean(all_gop_maps) if all_gop_maps else 0.0,
            'overall_mean_AP50': np.mean(all_gop_ap50s) if all_gop_ap50s else 0.0,
            'overall_mean_AP75': np.mean(all_gop_ap75s) if all_gop_ap75s else 0.0,
            'mAP_std': np.std(all_gop_maps) if all_gop_maps else 0.0,
            'consistency_rating': 'Consistent' if (np.std(all_gop_maps) < 0.1 if all_gop_maps else False) else 'Variable',
            'performance_level': self.get_performance_level(np.mean(all_gop_maps) if all_gop_maps else 0.0),
            'object_results': all_object_results
        }
        
        return video_metrics
    
    def get_performance_level(self, mean_map):
        """Get performance level based on mAP."""
        if mean_map >= 0.7:
            return "Excellent"
        elif mean_map >= 0.5:
            return "Good"
        elif mean_map >= 0.3:
            return "Fair"
        else:
            return "Poor"
    
    def save_video_results(self, video_metrics, video_info):
        """Save individual video results."""
        video_name = video_info['video_name']
        
        # Save JSON results
        json_path = self.results_dir / f"{video_name}_results.json"
        with open(json_path, 'w') as f:
            json.dump(video_metrics, f, indent=2, default=str)
        
        # Save CSV for object-level results
        if video_metrics['object_results']:
            df_objects = pd.DataFrame(video_metrics['object_results'])
            csv_path = self.results_dir / f"{video_name}_object_results.csv"
            df_objects.to_csv(csv_path, index=False)
        
        # Add to overall results
        self.all_results.append(video_metrics)
    
    def process_all_videos(self, num_gops=3):
        """Process all available MOT17 and MOT20 videos."""
        sequences = self.get_available_sequences()
        
        if not sequences:
            print("❌ No sequences found to process!")
            return
        
        print(f"\n🚀 Starting processing of {len(sequences)} sequences...")
        print(f"   GOPs per video: {num_gops}")
        print(f"   Output directory: {self.run_dir}")
        
        successful_videos = 0
        failed_videos = []
        
        for i, sequence in enumerate(sequences, 1):
            print(f"\n📋 Progress: {i}/{len(sequences)}")
            
            result = self.process_single_video(sequence, num_gops)
            if result:
                successful_videos += 1
            else:
                failed_videos.append(sequence)
        
        print(f"\n{'='*80}")
        print(f"🏁 PROCESSING COMPLETE")
        print(f"   ✅ Successful: {successful_videos}/{len(sequences)}")
        print(f"   ❌ Failed: {len(failed_videos)}")
        if failed_videos:
            print(f"   Failed videos: {failed_videos}")
        print(f"{'='*80}")
        
        # Generate comprehensive summary
        self.generate_comprehensive_summary()
    
    def generate_comprehensive_summary(self):
        """Generate comprehensive summary across all videos and datasets."""
        if not self.all_results:
            print("❌ No results to summarize!")
            return
        
        print(f"\n📊 Generating comprehensive summary...")
        
        # Organize by dataset
        dataset_results = {}
        for result in self.all_results:
            dataset = result['video_info']['dataset']
            if dataset not in dataset_results:
                dataset_results[dataset] = []
            dataset_results[dataset].append(result)
        
        # Calculate dataset-level statistics
        dataset_summary = {}
        all_video_data = []
        
        for dataset, videos in dataset_results.items():
            # Collect metrics
            dataset_maps = [v['overall_mean_mAP'] for v in videos]
            dataset_ap50s = [v['overall_mean_AP50'] for v in videos]
            dataset_ap75s = [v['overall_mean_AP75'] for v in videos]
            total_objects = sum(v['total_objects_tracked'] for v in videos)
            
            dataset_summary[dataset] = {
                'num_videos': len(videos),
                'total_objects_tracked': total_objects,
                'mean_mAP': np.mean(dataset_maps),
                'std_mAP': np.std(dataset_maps),
                'mean_AP50': np.mean(dataset_ap50s),
                'mean_AP75': np.mean(dataset_ap75s),
                'best_video': max(videos, key=lambda x: x['overall_mean_mAP']),
                'worst_video': min(videos, key=lambda x: x['overall_mean_mAP']),
                'video_results': videos
            }
            
            # Add to overall video data
            for video in videos:
                video_data = {
                    'dataset': dataset,
                    'video_name': video['video_info']['video_name'],
                    'video_id': video['video_info']['video_id'],
                    'mean_mAP': video['overall_mean_mAP'],
                    'mean_AP50': video['overall_mean_AP50'],
                    'mean_AP75': video['overall_mean_AP75'],
                    'objects_tracked': video['total_objects_tracked'],
                    'gops_processed': video['num_gops_processed'],
                    'performance_level': video['performance_level'],
                    'consistency': video['consistency_rating']
                }
                all_video_data.append(video_data)
        
        # Save comprehensive results
        self.save_comprehensive_summary(dataset_summary, all_video_data)
        
        # Print summary
        self.print_summary(dataset_summary)
    
    def save_comprehensive_summary(self, dataset_summary, all_video_data):
        """Save comprehensive summary to files."""
        
        # Save dataset summary JSON
        summary_json = {
            'timestamp': datetime.now().isoformat(),
            'total_videos': len(self.all_results),
            'datasets': {}
        }
        
        for dataset, summary in dataset_summary.items():
            summary_json['datasets'][dataset] = {
                'num_videos': summary['num_videos'],
                'total_objects_tracked': summary['total_objects_tracked'],
                'mean_mAP': summary['mean_mAP'],
                'std_mAP': summary['std_mAP'],
                'mean_AP50': summary['mean_AP50'],
                'mean_AP75': summary['mean_AP75'],
                'best_video': {
                    'name': summary['best_video']['video_info']['video_name'],
                    'mAP': summary['best_video']['overall_mean_mAP']
                },
                'worst_video': {
                    'name': summary['worst_video']['video_info']['video_name'],
                    'mAP': summary['worst_video']['overall_mean_mAP']
                }
            }
        
        # Save JSON summary
        with open(self.summary_dir / "dataset_summary.json", 'w') as f:
            json.dump(summary_json, f, indent=2)
        
        # Save detailed video results CSV
        df_videos = pd.DataFrame(all_video_data)
        df_videos.to_csv(self.summary_dir / "all_videos_summary.csv", index=False)
        
        # Save detailed object results CSV
        all_objects = []
        for result in self.all_results:
            for obj in result['object_results']:
                obj_data = obj.copy()
                obj_data['dataset'] = result['video_info']['dataset']
                obj_data['video_name'] = result['video_info']['video_name']
                obj_data['video_id'] = result['video_info']['video_id']
                all_objects.append(obj_data)
        
        if all_objects:
            df_objects = pd.DataFrame(all_objects)
            df_objects.to_csv(self.summary_dir / "all_objects_summary.csv", index=False)
        
        # Generate performance analysis
        self.generate_performance_analysis(dataset_summary, all_video_data)
        
        print(f"✅ Summary saved to: {self.summary_dir}")
    
    def generate_performance_analysis(self, dataset_summary, all_video_data):
        """Generate detailed performance analysis."""
        analysis = []
        
        analysis.append("COMPREHENSIVE PERFORMANCE ANALYSIS")
        analysis.append("="*80)
        analysis.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        analysis.append(f"Total Videos Processed: {len(all_video_data)}")
        analysis.append("")
        
        # Overall statistics
        all_maps = [v['mean_mAP'] for v in all_video_data]
        analysis.append("OVERALL STATISTICS:")
        analysis.append(f"  Mean mAP across all videos: {np.mean(all_maps):.3f} ± {np.std(all_maps):.3f}")
        analysis.append(f"  Best performing video: {max(all_video_data, key=lambda x: x['mean_mAP'])['video_name']} (mAP: {max(all_maps):.3f})")
        analysis.append(f"  Worst performing video: {min(all_video_data, key=lambda x: x['mean_mAP'])['video_name']} (mAP: {min(all_maps):.3f})")
        analysis.append("")
        
        # Dataset comparison
        analysis.append("DATASET COMPARISON:")
        for dataset, summary in dataset_summary.items():
            analysis.append(f"  {dataset}:")
            analysis.append(f"    Videos: {summary['num_videos']}")
            analysis.append(f"    Objects tracked: {summary['total_objects_tracked']}")
            analysis.append(f"    Mean mAP: {summary['mean_mAP']:.3f} ± {summary['std_mAP']:.3f}")
            analysis.append(f"    Mean AP@0.5: {summary['mean_AP50']:.3f}")
            analysis.append(f"    Mean AP@0.75: {summary['mean_AP75']:.3f}")
            analysis.append(f"    Best video: {summary['best_video']['video_info']['video_name']} (mAP: {summary['best_video']['overall_mean_mAP']:.3f})")
            analysis.append(f"    Worst video: {summary['worst_video']['video_info']['video_name']} (mAP: {summary['worst_video']['overall_mean_mAP']:.3f})")
            analysis.append("")
        
        # Performance level distribution
        performance_dist = {}
        for video in all_video_data:
            level = video['performance_level']
            performance_dist[level] = performance_dist.get(level, 0) + 1
        
        analysis.append("PERFORMANCE LEVEL DISTRIBUTION:")
        for level, count in performance_dist.items():
            percentage = (count / len(all_video_data)) * 100
            analysis.append(f"  {level}: {count} videos ({percentage:.1f}%)")
        analysis.append("")
        
        # Consistency analysis
        consistent_videos = sum(1 for v in all_video_data if v['consistency'] == 'Consistent')
        analysis.append("CONSISTENCY ANALYSIS:")
        analysis.append(f"  Consistent videos: {consistent_videos}/{len(all_video_data)} ({(consistent_videos/len(all_video_data)*100):.1f}%)")
        analysis.append("")
        
        # Save analysis
        with open(self.summary_dir / "performance_analysis.txt", 'w') as f:
            f.write('\n'.join(analysis))
    
    def print_summary(self, dataset_summary):
        """Print summary to console."""
        print(f"\n{'='*80}")
        print("📊 COMPREHENSIVE EVALUATION SUMMARY")
        print(f"{'='*80}")
        
        for dataset, summary in dataset_summary.items():
            print(f"\n🎯 {dataset} Dataset:")
            print(f"   Videos processed: {summary['num_videos']}")
            print(f"   Total objects tracked: {summary['total_objects_tracked']}")
            print(f"   Mean mAP: {summary['mean_mAP']:.3f} ± {summary['std_mAP']:.3f}")
            print(f"   Mean AP@0.5: {summary['mean_AP50']:.3f}")
            print(f"   Mean AP@0.75: {summary['mean_AP75']:.3f}")
            print(f"   Best: {summary['best_video']['video_info']['video_name']} (mAP: {summary['best_video']['overall_mean_mAP']:.3f})")
            print(f"   Worst: {summary['worst_video']['video_info']['video_name']} (mAP: {summary['worst_video']['overall_mean_mAP']:.3f})")
        
        print(f"\n📁 All results saved to: {self.run_dir}")
        print(f"   📹 Videos: {self.videos_dir}")
        print(f"   📊 Individual results: {self.results_dir}")
        print(f"   📈 Summary: {self.summary_dir}")

def main():
    """Main function to process all datasets."""
    print("🚀 MOT17 & MOT20 Comprehensive Evaluation")
    print("="*80)
    
    # Create processor
    processor = DatasetProcessor()
    
    # Process all videos
    processor.process_all_videos(num_gops=3)
    
    print("\n✅ All processing complete!")
    print(f"📁 Results saved to: {processor.run_dir}")

if __name__ == "__main__":
    main()
