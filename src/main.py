"""
Feature Extraction Pipeline for Driver Fatigue Detection
Goal: Transform raw driving video data into structured signals 
      showing signs of driver fatigue and stress
Pipeline: Spatial (frame-based) + Temporal (time-based) features
"""

import os
import random
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance
from collections import deque
from datetime import datetime

class FeatureExtractionPipeline:
	"""
	Complete feature extraction pipeline for driver fatigue detection
	Implements: Facial Landmarks, Temporal Features, Motion & Pose, Feature Vector
	"""
	
	def __init__(self, temporal_window=90):  # 3 seconds at 30fps
		# Initialize MediaPipe Face Mesh for facial landmarks
		self.mp_face_mesh = mp.solutions.face_mesh
		self.face_mesh = self.mp_face_mesh.FaceMesh(
			max_num_faces=1,
			refine_landmarks=True,
			min_detection_confidence=0.5,
			min_tracking_confidence=0.5
		)
		
		# Temporal window for feature tracking (2-6 seconds recommended)
		self.temporal_window = temporal_window
		self.ear_history = deque(maxlen=temporal_window)
		self.mar_history = deque(maxlen=temporal_window)
		
		# Eye and mouth landmark indices for MediaPipe
		self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
		self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
		self.MOUTH = [61, 291, 0, 17, 269, 405]
		
	def calculate_ear(self, eye_landmarks):
		"""
		Eye Aspect Ratio (EAR): Detects blinks and eye closure
		EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
		Lower EAR → eyes closing/closed
		"""
		# Vertical distances
		A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
		B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
		# Horizontal distance
		C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
		
		ear = (A + B) / (2.0 * C)
		return ear
	
	def calculate_mar(self, mouth_landmarks):
		"""
		Mouth Aspect Ratio (MAR): Detects yawns
		MAR = ||p2-p8|| / ||p1-p5||
		Higher MAR → mouth opening (yawn)
		"""
		# Vertical distance
		A = distance.euclidean(mouth_landmarks[1], mouth_landmarks[5])
		# Horizontal distance
		B = distance.euclidean(mouth_landmarks[0], mouth_landmarks[3])
		
		mar = A / B if B > 0 else 0
		return mar
	
	def estimate_head_pose(self, landmarks, frame_shape):
		"""
		Head Pose Estimation: Detect yaw, pitch, roll
		Detects nodding or looking away (distraction)
		"""
		h, w = frame_shape[:2]
		
		# 3D model points
		model_points = np.array([
			(0.0, 0.0, 0.0),             # Nose tip
			(0.0, -330.0, -65.0),        # Chin
			(-225.0, 170.0, -135.0),     # Left eye left corner
			(225.0, 170.0, -135.0),      # Right eye right corner
			(-150.0, -150.0, -125.0),    # Left Mouth corner
			(150.0, -150.0, -125.0)      # Right mouth corner
		])
		
		# 2D image points from landmarks
		image_points = np.array([
			landmarks[1],    # Nose tip
			landmarks[152],  # Chin
			landmarks[226],  # Left eye left corner
			landmarks[446],  # Right eye right corner
			landmarks[57],   # Left mouth corner
			landmarks[287]   # Right mouth corner
		], dtype="double")
		
		# Camera internals
		focal_length = w
		center = (w/2, h/2)
		camera_matrix = np.array([
			[focal_length, 0, center[0]],
			[0, focal_length, center[1]],
			[0, 0, 1]
		], dtype="double")
		
		dist_coeffs = np.zeros((4, 1))
		
		# Solve PnP
		success, rotation_vec, translation_vec = cv2.solvePnP(
			model_points, image_points, camera_matrix, dist_coeffs, 
			flags=cv2.SOLVEPNP_ITERATIVE
		)
		
		# Convert rotation vector to euler angles
		rotation_mat, _ = cv2.Rodrigues(rotation_vec)
		pose_mat = cv2.hconcat((rotation_mat, translation_vec))
		_, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
		
		pitch, yaw, roll = euler_angles.flatten()[:3]
		
		return pitch, yaw, roll
	
	def calculate_perclos(self, ear_threshold=0.2):
		"""
		PERCLOS: Percentage of Eye Closure
		Classic fatigue signal - % of time eyes are closed
		"""
		if len(self.ear_history) == 0:
			return 0.0
		
		closed_frames = sum(1 for ear in self.ear_history if ear < ear_threshold)
		perclos = (closed_frames / len(self.ear_history)) * 100
		return perclos
	
	def extract_features(self, frame):
		"""
		Extract all features from a single frame
		Returns: feature dictionary with spatial and temporal features
		"""
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		results = self.face_mesh.process(rgb_frame)
		
		features = {
			'ear_left': 0.0,
			'ear_right': 0.0,
			'ear_avg': 0.0,
			'mar': 0.0,
			'pitch': 0.0,
			'yaw': 0.0,
			'roll': 0.0,
			'perclos': 0.0,
			'face_detected': False
		}
		
		if results.multi_face_landmarks:
			face_landmarks = results.multi_face_landmarks[0]
			h, w = frame.shape[:2]
			
			# Convert landmarks to numpy array
			landmarks = np.array([
				[lm.x * w, lm.y * h] 
				for lm in face_landmarks.landmark
			])
			
			# 1. Calculate EAR (Eye Aspect Ratio)
			left_eye = landmarks[self.LEFT_EYE]
			right_eye = landmarks[self.RIGHT_EYE]
			
			ear_left = self.calculate_ear(left_eye)
			ear_right = self.calculate_ear(right_eye)
			ear_avg = (ear_left + ear_right) / 2.0
			
			# 2. Calculate MAR (Mouth Aspect Ratio)
			mouth = landmarks[self.MOUTH]
			mar = self.calculate_mar(mouth)
			
			# 3. Head Pose Estimation
			pitch, yaw, roll = self.estimate_head_pose(landmarks, frame.shape)
			
			# Update temporal history
			self.ear_history.append(ear_avg)
			self.mar_history.append(mar)
			
			# 4. Calculate PERCLOS
			perclos = self.calculate_perclos()
			
			features.update({
				'ear_left': ear_left,
				'ear_right': ear_right,
				'ear_avg': ear_avg,
				'mar': mar,
				'pitch': pitch,
				'yaw': yaw,
				'roll': roll,
				'perclos': perclos,
				'face_detected': True
			})
		
		return features
	
	def process_video(self, video_path, max_frames=300, output_file=None):
		"""
		Process video and extract features from frames
		Returns: tuple of (features_list, summary_dict)
		"""
		cap = cv2.VideoCapture(video_path)
		fps = cap.get(cv2.CAP_PROP_FPS)
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		
		features_list = []
		frame_count = 0
		
		msg = f"   Processing: {os.path.basename(video_path)}\n"
		msg += f"   FPS: {fps:.1f} | Total Frames: {total_frames} | Duration: {total_frames/fps:.1f}s\n"
		print(msg, end='')
		if output_file:
			output_file.write(msg)
		
		while cap.isOpened() and frame_count < max_frames:
			ret, frame = cap.read()
			if not ret:
				break
			
			features = self.extract_features(frame)
			features_list.append(features)
			frame_count += 1
		
		cap.release()
		
		# Calculate summary statistics
		valid_features = [f for f in features_list if f['face_detected']]
		
		summary = {
			'video_path': video_path,
			'video_name': os.path.basename(video_path),
			'fps': fps,
			'total_frames': total_frames,
			'duration': total_frames/fps,
			'processed_frames': frame_count,
			'valid_frames': len(valid_features),
			'avg_ear': 0.0,
			'avg_mar': 0.0,
			'avg_perclos': 0.0,
			'avg_yaw': 0.0,
			'fatigue_score': 0,
			'risk_level': 'UNKNOWN',
			'warnings': []
		}
		
		if valid_features:
			avg_ear = np.mean([f['ear_avg'] for f in valid_features])
			avg_mar = np.mean([f['mar'] for f in valid_features])
			avg_perclos = np.mean([f['perclos'] for f in valid_features])
			avg_yaw = np.mean([f['yaw'] for f in valid_features])
			
			summary.update({
				'avg_ear': avg_ear,
				'avg_mar': avg_mar,
				'avg_perclos': avg_perclos,
				'avg_yaw': avg_yaw
			})
			
			msg = f"   Processed {len(valid_features)}/{frame_count} frames with face detected\n"
			msg += f"   Avg EAR: {avg_ear:.3f} | Avg MAR: {avg_mar:.3f} | PERCLOS: {avg_perclos:.1f}%\n"
			msg += f"   Avg Head Yaw: {avg_yaw:.1f}° (looking {'left' if avg_yaw < -10 else 'right' if avg_yaw > 10 else 'forward'})\n"
			print(msg, end='')
			if output_file:
				output_file.write(msg)
			
			# Fatigue indicators
			fatigue_score = 0
			warnings = []
			
			if avg_ear < 0.25:
				fatigue_score += 1
				warning = f"   Low EAR detected - possible drowsiness\n"
				warnings.append("Low EAR - possible drowsiness")
				print(warning, end='')
				if output_file:
					output_file.write(warning)
			
			if avg_mar > 0.6:
				fatigue_score += 1
				warning = f"   High MAR detected - frequent yawning\n"
				warnings.append("High MAR - frequent yawning")
				print(warning, end='')
				if output_file:
					output_file.write(warning)
			
			if avg_perclos > 15:
				fatigue_score += 1
				warning = f"   High PERCLOS - eyes closed {avg_perclos:.1f}% of time\n"
				warnings.append(f"High PERCLOS - eyes closed {avg_perclos:.1f}% of time")
				print(warning, end='')
				if output_file:
					output_file.write(warning)
			
			if abs(avg_yaw) > 20:
				fatigue_score += 1
				warning = f"   Head pose deviation - possible distraction\n"
				warnings.append("Head pose deviation - possible distraction")
				print(warning, end='')
				if output_file:
					output_file.write(warning)
			
			risk_level = 'HIGH' if fatigue_score >= 3 else 'MODERATE' if fatigue_score >= 2 else 'LOW'
			
			summary.update({
				'fatigue_score': fatigue_score,
				'risk_level': risk_level,
				'warnings': warnings
			})
			
			if fatigue_score == 0:
				msg = f"   No significant fatigue indicators detected\n"
				print(msg, end='')
				if output_file:
					output_file.write(msg)
			else:
				msg = f"   Fatigue Score: {fatigue_score}/4 - {risk_level} RISK\n"
				print(msg, end='')
				if output_file:
					output_file.write(msg)
		else:
			msg = f"   No face detected in video\n"
			print(msg, end='')
			if output_file:
				output_file.write(msg)
		
		print()
		if output_file:
			output_file.write("\n")
		
		return features_list, summary


def demo_feature_extraction():
	"""
	Demo: Randomly select 3-5 videos and demonstrate feature extraction pipeline
	Results are saved to a timestamped output file
	"""
	# Create output filename with timestamp
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	output_filename = f"demo_results_{timestamp}.txt"
	output_path = os.path.join("results", output_filename)
	
	# Create results directory if it doesn't exist
	os.makedirs("results", exist_ok=True)
	
	# Open output file
	with open(output_path, 'w', encoding='utf-8') as output_file:
		header = "="*80 + "\n"
		header += "DRIVER FATIGUE DETECTION - FEATURE EXTRACTION PIPELINE DEMO\n"
		header += "="*80 + "\n"
		header += f"Demo Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
		header += "\nPipeline Components:\n"
		header += "1. Facial Landmarks: EAR (eye aspect ratio), MAR (mouth aspect ratio)\n"
		header += "2. Temporal Features: PERCLOS (% eyes closed over time)\n"
		header += "3. Motion & Pose: Head pose estimation (yaw, pitch, roll)\n"
		header += "4. Feature Vector: Combined normalized features for ML model\n"
		header += "\n" + "-"*80 + "\n\n"
		
		print(header, end='')
		output_file.write(header)
		
		# Find all available videos by scanning the configured raw data path recursively
		try:
			from config import DATA_RAW_PATH
		except Exception:
			DATA_RAW_PATH = 'data/raw'

		raw_root = os.path.abspath(DATA_RAW_PATH)
		all_videos = []
		for root, _, files in os.walk(raw_root):
			for f in files:
				if f.lower().endswith('.mp4'):
					all_videos.append(os.path.join(root, f))
		
		if not all_videos:
			msg = "No videos found in dataset\n"
			print(msg, end='')
			output_file.write(msg)
			return
		
		# Randomly select 3-5 videos (or fewer if dataset is small)
		sample_size = random.randint(3, 5)
		demo_videos = random.sample(all_videos, min(sample_size, len(all_videos))) if all_videos else []
		
		msg = f"Randomly Selected {len(demo_videos)} Videos for Demo:\n\n"
		for idx, video in enumerate(demo_videos, 1):
			msg += f"{idx}. {video}\n"
		
		msg += "EXTRACTING FEATURES...\n"
		
		print(msg, end='')
		output_file.write(msg)
		
		# Initialize pipeline
		pipeline = FeatureExtractionPipeline()
		
		# Store all summaries
		all_summaries = []
		
		# Process each video
		for idx, video_path in enumerate(demo_videos, 1):
			msg = f"Video {idx}/{len(demo_videos)}:\n"
			msg += "-" * 80 + "\n"
			print(msg, end='')
			output_file.write(msg)
			
			features, summary = pipeline.process_video(video_path, max_frames=150, output_file=output_file)
			all_summaries.append(summary)
		
		# Write summary section
		summary_header = "\n" + "="*80 + "\n"
		summary_header += "DEMO SUMMARY\n"
		summary_header += "="*80 + "\n\n"
		
		print(summary_header, end='')
		output_file.write(summary_header)
		
		# Overall statistics
		total_videos = len(all_summaries)
		high_risk = sum(1 for s in all_summaries if s['risk_level'] == 'HIGH')
		moderate_risk = sum(1 for s in all_summaries if s['risk_level'] == 'MODERATE')
		low_risk = sum(1 for s in all_summaries if s['risk_level'] == 'LOW')
		
		summary_stats = f"Total Videos Processed: {total_videos}\n"
		summary_stats += f" HIGH RISK: {high_risk}\n"
		summary_stats += f" MODERATE RISK: {moderate_risk}\n"
		summary_stats += f" LOW RISK: {low_risk}\n\n"
		
		summary_stats += "Individual Video Results:\n"
		summary_stats += "-" * 80 + "\n"
		
		for idx, summary in enumerate(all_summaries, 1):
			summary_stats += f"\n{idx}. {summary['video_name']}\n"
			summary_stats += f"   Duration: {summary['duration']:.1f}s | Frames: {summary['valid_frames']}/{summary['processed_frames']}\n"
			summary_stats += f"   EAR: {summary['avg_ear']:.3f} | MAR: {summary['avg_mar']:.3f} | PERCLOS: {summary['avg_perclos']:.1f}%\n"
			summary_stats += f"   Head Yaw: {summary['avg_yaw']:.1f}°\n"
			summary_stats += f"   Risk Level: {summary['risk_level']} (Score: {summary['fatigue_score']}/4)\n"
			if summary['warnings']:
				summary_stats += f"   Warnings: {', '.join(summary['warnings'])}\n"
		
		print(summary_stats, end='')
		output_file.write(summary_stats)
	
	# Final message with output file location
	final_msg = f"\nResults saved to: {output_path}\n"
	print(final_msg)
	
	return output_path

def _orchestrate(args=None):
	import argparse
	from pathlib import Path
	import config

	ap = argparse.ArgumentParser(description="Orchestrate pipeline steps: build-index, preprocess, demo, train, eval")
	ap.add_argument('--build-index', action='store_true', help='Build dataset_index.csv from data/raw')
	ap.add_argument('--preprocess', action='store_true', help='Build sliding-window .npz files')
	ap.add_argument('--demo', action='store_true', help='Run the demo feature extraction and save report')
	ap.add_argument('--train', action='store_true', help='Run training script (may be slow)')
	ap.add_argument('--eval', action='store_true', help='Run evaluation script on checkpoint')
	ap.add_argument('--all', action='store_true', help='Run build-index, preprocess, and demo (default if no flags)')
	parsed = ap.parse_args(args=args)

	# Default behavior: if no flags provided, run build-index + preprocess + demo
	if not any([parsed.build_index, parsed.preprocess, parsed.demo, parsed.train, parsed.eval, parsed.all]):
		parsed.all = True

	project_data_dir = Path(config.DATA_RAW_PATH).parent

	if parsed.build_index or parsed.all:
		print('[ORCH] Building dataset index...')
		try:
			# Download datasets from Kaggle if needed
			try:
				from src.data.kaggle_fetch import download_datasets
				from config import DATA_RAW_PATH
				download_datasets(DATA_RAW_PATH)
			except Exception as e:
				print(f"Could not download datasets: {e}")

			import data
			data.run(project_data_dir, do_standardize=False)
		except Exception as e:
			print(f'[ERROR] build-index failed: {e}')

	if parsed.preprocess or parsed.all:
		print('[ORCH] Running preprocessing (sliding windows)...')
		try:
			import preprocess
			index_csv = project_data_dir / 'processed' / 'dataset_index.csv'
			out_dir = project_data_dir / 'processed' / 'windows'
			# Optional environment variables to speed up preprocessing for debugging
			sample_rate = int(os.environ.get('PREPROCESS_SAMPLE_RATE', '1'))
			max_files = os.environ.get('PREPROCESS_MAX_FILES')
			max_files = int(max_files) if max_files is not None else None
			preprocess.build_windows(index_csv, out_dir, win_sec=config.WIN_SEC, stride_sec=config.STRIDE_SEC, img_size=224, sample_rate=sample_rate, max_files=max_files)
		except Exception as e:
			print(f'[ERROR] preprocess failed: {e}')

	if parsed.demo or parsed.all:
		print('[ORCH] Running demo...')
		demo_feature_extraction()

	if parsed.train:
		print('[ORCH] Launching training for Model 1 (ResNet-18 + LSTM + NAT)...')
		import subprocess
		import sys
		# Determine which windows directory to use (prefer windows_bench_mp if it exists)
		windows_dir = project_data_dir / 'processed' / 'windows_bench_mp'
		if not windows_dir.exists():
			windows_dir = project_data_dir / 'processed' / 'windows_bench_serial'
		if not windows_dir.exists():
			windows_dir = project_data_dir / 'processed' / 'windows'
		
		print(f'[ORCH] Using windows directory: {windows_dir}')
		
		# Use module invocation to ensure correct path handling
		cmd = [
			sys.executable, '-u', '-m', 'src.train',
			'--splits_csv', str(project_data_dir / 'processed' / 'dataset_index.csv'),
			'--out_dir', 'outputs',
			'--epochs', '10',
			'--batch_size', '2',  # Reduced batch size for stability
			'--lr', '1e-4',
			'--num_classes', '8',  # 8 classes from LABEL_TO_INT mapping
			'--seq_len', '40',  # Match average window length (~38 frames)
			'--img_size', '224',
			'--backbone', 'resnet18',
			'--use_nat',  # Enable Neighborhood Attention
			'--nat_kernel', '7',
			'--nat_blocks', '2',
			'--num_workers', '0'  # Use 0 workers to avoid shared memory issues
		]
		print(f'[ORCH] Running: {" ".join(cmd)}')
		subprocess.run(cmd, check=False)

	if parsed.eval:
		print('[ORCH] Running evaluation (via subprocess)...')
		import subprocess
		ckpt = 'outputs/best.pt'
		cmd = ['python', '-u', '-m', 'src.eval', '--splits_csv', str(project_data_dir / 'processed' / 'dataset_index.csv'), '--ckpt', ckpt]
		subprocess.run(cmd, check=False)


if __name__ == "__main__":
	_orchestrate()