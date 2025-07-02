"""
Video frame extraction utility for camera trap footage preprocessing.

Batch processes video files to extract frames at regular time intervals, creating 
individual image files suitable for further analysis with detection/classification 
pipelines. Designed for processing camera trap videos into manageable frame datasets.

Key Features:
- Recursive directory scanning for multiple video formats (mp4, avi, mov, mkv, wmv, flv)
- Configurable time intervals between extracted frames
- Organized output structure with separate folders per video
- Frame numbering with zero-padding for proper sorting
- Video metadata extraction (FPS, duration, total frames)

Output Structure:
- output_dir/[video_name]/[video_name]_frame_0001.jpg
- output_dir/[video_name]/[video_name]_frame_0002.jpg
- etc.

Usage Examples:
- python video_to_frames.py /path/to/videos /path/to/frames --interval 5.0
- python video_to_frames.py /videos /frames --interval 1.0 --recursive

Typical workflow: Video → Frames → Detection pipeline → Classification
Essential for converting camera trap videos into frame-based datasets for analysis.
"""

import os
import argparse
import cv2
from pathlib import Path

def extract_frames(video_path, output_dir, interval_seconds):
    """
    Extract frames from a video at specified time intervals.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save the extracted frames
        interval_seconds (float): Time interval between frames in seconds
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # Calculate frame interval
    frame_interval = int(fps * interval_seconds)
    
    # Get the video filename without extension
    video_name = Path(video_path).stem
    
    # Extract frames
    count = 0
    frame_count = 0
    
    while True:
        ret, frame = video.read()
        
        if not ret:
            break
            
        if count % frame_interval == 0:
            # Save the frame
            output_path = os.path.join(output_dir, f"{video_name}_frame_{frame_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            frame_count += 1
            
        count += 1
    
    # Release the video capture object
    video.release()
    
    print(f"Extracted {frame_count} frames from {video_path} (duration: {duration:.2f}s)")

def find_video_files(directory):
    """
    Find all video files in a directory.
    
    Args:
        directory (str): Directory to search for videos
        
    Returns:
        list: List of video file paths
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    video_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))
    
    return video_files

def main():
    parser = argparse.ArgumentParser(description='Extract frames from videos at regular intervals')
    parser.add_argument('input_dir', help='Directory containing video files')
    parser.add_argument('output_dir', help='Directory to save extracted frames')
    parser.add_argument('--interval', type=float, default=1.0, 
                        help='Time interval between frames in seconds (default: 1.0)')
    parser.add_argument('--recursive', action='store_true', 
                        help='Search for videos recursively in subdirectories')
    
    args = parser.parse_args()
    
    # Find all video files
    video_files = find_video_files(args.input_dir)
    
    if not video_files:
        print(f"No video files found in {args.input_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Process each video
    for video_path in video_files:
        # Create a subdirectory for each video
        video_name = Path(video_path).stem
        video_output_dir = os.path.join(args.output_dir, video_name)
        
        print(f"Processing: {video_path}")
        extract_frames(video_path, video_output_dir, args.interval)

if __name__ == '__main__':
    main()