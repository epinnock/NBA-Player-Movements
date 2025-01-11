import cv2
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import json
import logging
import tempfile
import shutil
import time
import os
import re
from datetime import datetime
from filelock import FileLock
import atexit
from contextlib import contextmanager
from functools import lru_cache

class FrameProcessingError(Exception):
    """Custom exception for frame processing errors"""
    pass

class ManifestError(Exception):
    """Custom exception for manifest-related errors"""
    pass

class VideoFormatError(Exception):
    """Custom exception for video format validation errors"""
    pass

class FrameChangeFinder:
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mkv', '.mov'}
    
    def __init__(self, video_path: str, output_dir: Optional[str] = None, 
                 max_offset: int = 3, chunk_size: int = 30, max_memory_entries: int = 1000):
        """
        Initialize the FrameChangeFinder with enhanced error handling and configuration.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory for output files (default: 'processed_frames')
            max_offset: Maximum frames to shift when a frame fails (default: 3)
            chunk_size: Size of frame chunks for processing (default: 30)
            max_memory_entries: Maximum number of frame entries to keep in memory (default: 1000)
        
        Raises:
            VideoFormatError: If video format is not supported
            FrameProcessingError: If video file cannot be opened
        """
        self.video_path = Path(video_path)
        self._validate_video_format()
        
        # Initialize video capture with error checking
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise FrameProcessingError(f"Failed to open video file: {video_path}")
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames <= 0:
            raise FrameProcessingError(f"Invalid frame count: {self.total_frames}")
        
        # Configuration
        self.MAX_OFFSET = max_offset
        self.CHUNK_SIZE = chunk_size
        self.MAX_MEMORY_ENTRIES = max_memory_entries
        
        # Set up directories
        self.output_dir = Path(output_dir) if output_dir else Path('processed_frames')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.output_dir / 'frames'
        self.frames_dir.mkdir(exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Set up logging
        self.setup_logging()
        
        # Initialize manifest and lock
        self.manifest_path = self.output_dir / 'manifest.json'
        self.lock_path = self.output_dir / 'manifest.lock'
        self.lock = FileLock(str(self.lock_path))
        
        # Cache for processed frames
        self._frame_cache = {}
        self._manifest_modified = False
        
        # Initialize manifest
        self.initialize_manifest()
        
        # Register cleanup handler
        atexit.register(self.cleanup)
        
        logging.info(f"Initialized FrameChangeFinder for {video_path}")
        logging.info(f"Total frames: {self.total_frames}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup"""
        self.cleanup()
        return False  # Don't suppress exceptions
    
    def setup_logging(self):
        """Set up logging configuration"""
        log_file = self.output_dir / f"frame_processing_{datetime.now():%Y%m%d_%H%M%S}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(str(log_file)),
                logging.StreamHandler()
            ]
        )

    def _validate_video_format(self):
        """Validate video file format"""
        if not self.video_path.exists():
            raise VideoFormatError(f"Video file not found: {self.video_path}")
        
        if self.video_path.suffix.lower() not in self.SUPPORTED_VIDEO_FORMATS:
            raise VideoFormatError(
                f"Unsupported video format: {self.video_path.suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_VIDEO_FORMATS)}"
            )
    
    @contextmanager
    def manifest_transaction(self):
        """Context manager for manifest transactions"""
        try:
            with self.lock:
                yield
                if self._manifest_modified:
                    self.save_manifest_safely()
                    self._manifest_modified = False
        except Exception as e:
            logging.error(f"Transaction error: {str(e)}")
            raise

    def initialize_manifest(self):
        """Initialize or load the manifest file with proper error handling"""
        try:
            with self.lock:
                if self.manifest_path.exists():
                    # Load existing manifest
                    with open(self.manifest_path, 'r') as f:
                        self.manifest = json.load(f)
                    if not self.validate_manifest(self.manifest):
                        backup_path = self.manifest_path.with_suffix(f'.bak.{int(time.time())}')
                        shutil.copy2(self.manifest_path, backup_path)
                        logging.warning(f"Invalid manifest found, backed up to {backup_path}")
                        self.manifest = self.create_new_manifest()
                else:
                    self.manifest = self.create_new_manifest()
                
                # Initialize processed frames set
                self.processed_frames = set(
                    entry['frame_number'] 
                    for entry in self.manifest['processed_frames']
                )
                
                # Sort manifest entries
                self.manifest['processed_frames'] = sorted(
                    self.manifest['processed_frames'],
                    key=lambda x: x['frame_number']
                )
                self._manifest_modified = True
                self.save_manifest_safely()
        
        except Exception as e:
            logging.error(f"Error initializing manifest: {str(e)}")
            self.manifest = self.create_new_manifest()
            self.processed_frames = set()

    def create_new_manifest(self) -> dict:
        """Create a new manifest structure"""
        return {
            'video_path': str(self.video_path),
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'total_frames': self.total_frames,
            'processed_frames': []
        }

    def validate_manifest(self, manifest: dict) -> bool:
        """Validate manifest structure and content"""
        required_keys = {'video_path', 'created_at', 'last_updated', 'total_frames', 'processed_frames'}
        try:
            # Check basic structure
            if not all(key in manifest for key in required_keys):
                return False
            
            # Check processed frames entries
            for entry in manifest['processed_frames']:
                if not self.validate_frame_entry(entry):
                    return False
            
            return True
        except Exception as e:
            logging.error(f"Error validating manifest: {str(e)}")
            return False

    def validate_frame_entry(self, entry: dict) -> bool:
        """Validate a single frame entry in the manifest"""
        required_keys = {'frame_number', 'frame_path', 'data', 'is_change'}
        if not all(key in entry for key in required_keys):
            return False
        
        # Validate frame number
        if not isinstance(entry['frame_number'], int) or entry['frame_number'] < 0:
            return False
        
        # Validate frame path
        if not isinstance(entry['frame_path'], str):
            return False
        
        # Validate data structure
        if not self.validate_timestamp(entry['data']):
            return False
        
        return True

    def validate_timestamp(self, timestamp: dict) -> bool:
        """Validate timestamp format and values"""
        try:
            if not all(key in timestamp for key in ['quarter', 'gameclock', 'shotclock']):
                return False
            
            # Validate quarter
            if not isinstance(timestamp['quarter'], int) or timestamp['quarter'] < 1:
                return False
            
            # Validate gameclock format (MM:SS)
            if not re.match(r'^\d{1,2}:\d{2}$', timestamp['gameclock']):
                return False
            
            # Validate shotclock (two digits, can start with 0)
            if not re.match(r'^\d{2}$', str(timestamp['shotclock'])):
                return False
            
            return True
        except Exception as e:
            logging.error(f"Error validating timestamp: {str(e)}")
            return False

    def normalize_timestamp(self, timestamp: dict) -> dict:
        """Normalize timestamp format for consistent comparison"""
        try:
            # Ensure shotclock is two digits
            timestamp['shotclock'] = str(timestamp['shotclock']).zfill(2)
            
            # Normalize gameclock to MM:SS format
            if ':' in timestamp['gameclock']:
                minutes, seconds = timestamp['gameclock'].split(':')
                timestamp['gameclock'] = f"{int(minutes)}:{seconds.zfill(2)}"
            
            return timestamp
        except Exception as e:
            logging.error(f"Error normalizing timestamp: {str(e)}")
            return timestamp

    def timestamps_differ(self, timestamp1: Optional[dict], timestamp2: Optional[dict]) -> bool:
        """Compare two timestamps to check if clocks differ with improved validation"""
        if timestamp1 is None or timestamp2 is None:
            return False
            
        try:
            ts1 = self.normalize_timestamp(
                timestamp1 if isinstance(timestamp1, dict) else json.loads(timestamp1)
            )
            ts2 = self.normalize_timestamp(
                timestamp2 if isinstance(timestamp2, dict) else json.loads(timestamp2)
            )
            
            # Compare all clock values and log changes
            diffs = []
            if ts1['quarter'] != ts2['quarter']:
                diffs.append(f"quarter: {ts1['quarter']} -> {ts2['quarter']}")
            if ts1['gameclock'] != ts2['gameclock']:
                diffs.append(f"gameclock: {ts1['gameclock']} -> {ts2['gameclock']}")
            if ts1['shotclock'] != ts2['shotclock']:
                diffs.append(f"shotclock: {ts1['shotclock']} -> {ts2['shotclock']}")
            
            if diffs:
                logging.debug(f"Clock changes detected: {', '.join(diffs)}")
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"Error comparing timestamps: {str(e)}")
            return False

    def save_manifest_safely(self):
        """Save manifest with backup and atomic writing"""
        try:
            with self.lock:
                # Update timestamp
                self.manifest['last_updated'] = datetime.now().isoformat()
                
                # Create temporary file
                temp_path = self.manifest_path.with_suffix('.tmp')
                backup_path = self.manifest_path.with_suffix('.bak')
                
                # Write to temporary file
                with open(temp_path, 'w') as f:
                    json.dump(self.manifest, f, indent=4)
                
                # Create backup of current manifest if it exists
                if self.manifest_path.exists():
                    shutil.copy2(self.manifest_path, backup_path)
                
                # Atomic rename of temporary file to final manifest
                os.replace(temp_path, self.manifest_path)
                self._manifest_modified = False
                
        except Exception as e:
            logging.error(f"Error saving manifest: {str(e)}")
            raise ManifestError(f"Failed to save manifest: {str(e)}")

    @lru_cache(maxsize=1000)
    def process_frame_cached(self, frame_number: int, generate_json_timestamp) -> Optional[dict]:
        """Cached version of frame processing"""
        return self._process_frame_internal(frame_number, generate_json_timestamp)

    def process_frame(self, frame_number: int, generate_json_timestamp, is_change: bool = False) -> Optional[dict]:
        """Process a single frame with enhanced retry logic and validation"""
        # Check if frame was already processed
        if frame_number in self.processed_frames:
            # Update change status if needed
            if is_change:
                self.update_frame_change_status(frame_number, True)
            
            # Return existing timestamp
            for entry in self.manifest['processed_frames']:
                if entry['frame_number'] == frame_number:
                    return entry['data']
            return None
        
        # Process new frame
        for offset in range(self.MAX_OFFSET):
            try:
                current_frame = frame_number + offset
                if current_frame >= self.total_frames:
                    return None
                
                frame_path = self.save_frame(current_frame)
                if frame_path is None:
                    continue
                
                timestamp = generate_json_timestamp(frame_path)
                if timestamp is None or not self.validate_timestamp(timestamp):
                    continue
                
                # Save frame data
                self.save_processed_frame(current_frame, frame_path, timestamp, is_change)
                
                if offset > 0:
                    logging.info(f"Successfully processed frame {current_frame} (offset +{offset} from {frame_number})")
                return timestamp
                
            except Exception as e:
                logging.warning(f"Failed to process frame {current_frame}: {str(e)}")
                continue
        
        logging.error(f"Failed to process frame {frame_number} after {self.MAX_OFFSET} attempts")
        return None
    
    def _process_frame_internal(self, frame_number: int, generate_json_timestamp) -> Optional[dict]:
        """Internal frame processing logic"""
        if frame_number in self._frame_cache:
            return self._frame_cache[frame_number]
            
        frame_path = self.save_frame(frame_number)
        if frame_path is None:
            return None
            
        try:
            timestamp = generate_json_timestamp(frame_path)
            if timestamp and self.validate_timestamp(timestamp):
                self._frame_cache[frame_number] = timestamp
                return timestamp
        except Exception as e:
            logging.error(f"Error processing frame {frame_number}: {str(e)}")
            
        return None

    def save_frame(self, frame_number: int) -> Optional[str]:
        """Save a specific frame as image with enhanced error handling"""
        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if not ret:
                return None
            
            frame_path = str(self.temp_dir / f"frame_{frame_number:08d}.jpg")
            if not cv2.imwrite(frame_path, frame):
                raise FrameProcessingError(f"Failed to write frame {frame_number}")
            
            return frame_path
            
        except Exception as e:
            logging.error(f"Error saving frame {frame_number}: {str(e)}")
            return None

    def save_processed_frame(self, frame_number: int, frame_path: str, timestamp: dict, is_change: bool = False) -> bool:
        """Save processed frame with optimized manifest handling"""
        if frame_number in self.processed_frames:
            return True
            
        try:
            with self.manifest_transaction():
                # Copy frame to permanent storage
                permanent_frame_path = self.frames_dir / f"frame_{frame_number:08d}.jpg"
                shutil.copy2(frame_path, permanent_frame_path)
                
                # Add to manifest
                frame_entry = {
                    'frame_number': frame_number,
                    'frame_path': str(permanent_frame_path.relative_to(self.output_dir)),
                    'data': self.normalize_timestamp(timestamp),
                    'is_change': is_change
                }
                
                self.manifest['processed_frames'].append(frame_entry)
                self.processed_frames.add(frame_number)
                
                # Memory management
                if len(self.manifest['processed_frames']) > self.MAX_MEMORY_ENTRIES:
                    self.flush_memory()
                
                self._manifest_modified = True
                logging.info(f"Saved frame {frame_number} (change: {is_change})")
                return True
                
        except Exception as e:
            logging.error(f"Error saving processed frame {frame_number}: {str(e)}")
            return False

    def update_frame_change_status(self, frame_number: int, is_change: bool = True) -> bool:
        """Update the change status of a frame in the manifest"""
        try:
            with self.lock:
                for entry in self.manifest['processed_frames']:
                    if entry['frame_number'] == frame_number:
                        if entry['is_change'] != is_change:
                            entry['is_change'] = is_change
                            logging.info(f"Updated frame {frame_number} change status to {is_change}")
                            self._manifest_modified = True
                            self.save_manifest_safely()
                        return True
                return False
                
        except Exception as e:
            logging.error(f"Error updating frame status: {str(e)}")
            return False

    def find_change_point(self, start: int, end: int, generate_json_timestamp) -> Optional[int]:
        """Binary search to find frame where clock changes with enhanced validation"""
        if end <= start + 1:
            return None
            
        start_timestamp = self.process_frame_cached(start, generate_json_timestamp)
        end_timestamp = self.process_frame_cached(end, generate_json_timestamp)
        
        if start_timestamp is None or end_timestamp is None:
            return None
            
        if not self.timestamps_differ(start_timestamp, end_timestamp):
            return None
            
        if end == start + 2:
            middle_frame = start + 1
            middle_timestamp = self.process_frame_cached(middle_frame, generate_json_timestamp)
            
            if middle_timestamp is not None:
                # Verify which frame is the actual change point
                if self.timestamps_differ(start_timestamp, middle_timestamp):
                    if self.verify_change_point(middle_frame, generate_json_timestamp):
                        return middle_frame
                if self.timestamps_differ(middle_timestamp, end_timestamp):
                    if self.verify_change_point(end, generate_json_timestamp):
                        return end
            return end
            
        mid = (start + end) // 2
        mid_timestamp = self.process_frame_cached(mid, generate_json_timestamp)
        
        if mid_timestamp is None:
            # Try alternate mid point
            alt_mid = mid + 1
            mid_timestamp = self.process_frame_cached(alt_mid, generate_json_timestamp)
            if mid_timestamp is not None:
                mid = alt_mid
            else:
                return None
        
        # Search both halves if necessary
        if self.timestamps_differ(start_timestamp, mid_timestamp):
            change_point = self.find_change_point(start, mid, generate_json_timestamp)
            if change_point is not None:
                return change_point
        
        return self.find_change_point(mid, end, generate_json_timestamp)

    def verify_change_point(self, frame_number: int, generate_json_timestamp) -> bool:
        """Verify a change point by checking adjacent frames"""
        try:
            # Get timestamps for adjacent frames
            prev_ts = None if frame_number <= 0 else self.process_frame(frame_number - 1, generate_json_timestamp)
            curr_ts = self.process_frame(frame_number, generate_json_timestamp)
            next_ts = None if frame_number >= self.total_frames - 1 else self.process_frame(frame_number + 1, generate_json_timestamp)
            
            # Check for changes
            prev_change = prev_ts is not None and self.timestamps_differ(prev_ts, curr_ts)
            next_change = next_ts is not None and self.timestamps_differ(curr_ts, next_ts)
            
            return prev_change or next_change
            
        except Exception as e:
            logging.error(f"Error verifying change point at frame {frame_number}: {str(e)}")
            return False

    def flush_memory(self):
        """Improved memory management with LRU-style caching"""
        try:
            # Sort entries by importance (changes first, then most recent)
            entries = self.manifest['processed_frames']
            change_entries = [e for e in entries if e['is_change']]
            normal_entries = [e for e in entries if not e['is_change']]
            
            # Keep all change entries and most recent normal entries
            keep_normal = max(0, self.MAX_MEMORY_ENTRIES - len(change_entries))
            kept_entries = change_entries + normal_entries[-keep_normal:]
            
            # Update manifest and processed frames set
            self.manifest['processed_frames'] = sorted(kept_entries, key=lambda x: x['frame_number'])
            self.processed_frames = set(entry['frame_number'] for entry in kept_entries)
            
            # Clear frame cache
            self._frame_cache.clear()
            self.process_frame_cached.cache_clear()
            
            logging.info(
                f"Memory flushed. Keeping {len(change_entries)} change frames "
                f"and {len(kept_entries) - len(change_entries)} normal frames"
            )
            
        except Exception as e:
            logging.error(f"Error flushing memory: {str(e)}")

    def find_all_changes(self, generate_json_timestamp) -> List[Tuple[int, dict]]:
        """Find all frames where clock changes occur with enhanced validation and error handling"""
        changes = []
        errors = 0
        processed_chunks = 0
        
        # Try to resume from last processed frame
        current_frame = self.get_last_processed_frame()
        logging.info(f"Starting processing from frame {current_frame} of {self.total_frames} total frames")
        
        try:
            while current_frame < self.total_frames - self.CHUNK_SIZE:
                try:
                    end_frame = min(current_frame + self.CHUNK_SIZE, self.total_frames)
                    logging.info(f"Processing frames {current_frame}-{end_frame}")
                    
                    start_timestamp = self.process_frame(current_frame, generate_json_timestamp)
                    end_timestamp = self.process_frame(end_frame, generate_json_timestamp)
                    
                    if start_timestamp is None or end_timestamp is None:
                        shift = max(1, self.CHUNK_SIZE // 2)
                        current_frame += shift
                        logging.warning(f"Shifting window by {shift} frames due to processing failure")
                        errors += 1
                        continue
                    
                    if self.timestamps_differ(start_timestamp, end_timestamp):
                        change_frame = self.find_change_point(current_frame, end_frame, generate_json_timestamp)
                        if change_frame is not None and self.verify_change_point(change_frame, generate_json_timestamp):
                            # Process the change frame and mark it as a change
                            timestamp = self.process_frame(change_frame, generate_json_timestamp, is_change=True)
                            if timestamp is not None:
                                changes.append((change_frame, timestamp))
                                logging.info(f"Found verified change at frame {change_frame}")
                                
                                # Continue searching from the change point
                                current_frame = change_frame
                                continue
                    
                    current_frame = end_frame
                    processed_chunks += 1
                    
                    # Log progress
                    if processed_chunks % max(1, self.total_frames // (20 * self.CHUNK_SIZE)) == 0:
                        progress = (current_frame / self.total_frames) * 100
                        logging.info(f"Progress: {progress:.1f}% ({current_frame}/{self.total_frames} frames)")
                    
                except Exception as e:
                    logging.error(f"Error processing chunk at frame {current_frame}: {str(e)}")
                    current_frame += max(1, self.CHUNK_SIZE // 2)
                    errors += 1
            
        except KeyboardInterrupt:
            logging.warning("Processing interrupted by user")
        finally:
            self.cleanup()
            
            # Log final statistics
            processed_frames = len(self.processed_frames)
            success_rate = (processed_frames / self.total_frames * 100) if self.total_frames > 0 else 0
            logging.info(f"Processing completed:")
            logging.info(f"- Frames processed: {processed_frames}/{self.total_frames}")
            logging.info(f"- Success rate: {success_rate:.1f}%")
            logging.info(f"- Total errors: {errors}")
            logging.info(f"- Changes found: {len(changes)}")
        
        return sorted(changes)

    def cleanup(self):
        """Enhanced cleanup with proper resource management"""
        try:
            # Save final manifest state if modified
            if self._manifest_modified:
                with self.manifest_transaction():
                    pass  # Transaction will save if needed
            
            # Clear caches
            self._frame_cache.clear()
            self.process_frame_cached.cache_clear()
            
            # Release video capture
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
                self.cap = None
            
            # Clean up temporary directory
            if hasattr(self, 'temp_dir') and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
        finally:
            # Remove atexit handler
            try:
                atexit.unregister(self.cleanup)
            except Exception:
                pass

    def get_last_processed_frame(self) -> int:
        """Get the last successfully processed frame number"""
        try:
            if not self.manifest['processed_frames']:
                return 0
            return max(entry['frame_number'] for entry in self.manifest['processed_frames'])
        except Exception as e:
            logging.error(f"Error getting last processed frame: {str(e)}")
            return 0