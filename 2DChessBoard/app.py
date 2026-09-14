import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pyautogui  # For screen capture
from pathlib import Path
import time


class ChessPieceCNN(nn.Module):
    def __init__(self, num_classes):
        super(ChessPieceCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten for FC layers
        x = x.view(-1, 128 * 8 * 8)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


def find_chessboard_corners(img):
    """
    Takes a loaded image, finds the four corners of the largest contour,
    and returns them. Same code reused from process_data.py to maintain consistency
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blur, 50, 150)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv.contourArea)
    points = largest_contour.reshape(len(largest_contour), 2)
    corners = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    corners[0] = points[np.argmin(s)]  # Top-left
    corners[2] = points[np.argmax(s)]  # Bottom-right
    diff = np.diff(points, axis=1)
    corners[1] = points[np.argmin(diff)]  # Top-right
    corners[3] = points[np.argmax(diff)]  # Bottom-left

    return corners


def extract_chessboard(img, corners):
    """
    Applies perspective transform to extract the chessboard.
    Returns 800x800 warped board image.
    """
    dst_points = np.array([[0, 0], [799, 0], [799, 799], [0, 799]], dtype="float32")
    M = cv.getPerspectiveTransform(corners, dst_points)
    warped_board = cv.warpPerspective(img, M, (800, 800))
    return warped_board


def split_board_into_squares(warped_board):
    """
    Splits the 800x800 board into 64 squares (8x8).
    Returns a list of square images in row-major order (a8 to h1 for standard orientation).
    """
    square_size = 100
    squares = []

    for row in range(8):
        for col in range(8):
            square_img = warped_board[
                         row * square_size:(row + 1) * square_size,
                         col * square_size:(col + 1) * square_size
                         ]
            squares.append(square_img)

    return squares


def preprocess_square(square_img, target_size=(64, 64)):
    """
    Preprocesses a square image for model inference.
    Converts OpenCV BGR to RGB, resizes, and applies transformations.
    """
    # Convert BGR to RGB
    square_rgb = cv.cvtColor(square_img, cv.COLOR_BGR2RGB)

    # Resize if needed
    if square_rgb.shape[:2] != target_size:
        square_rgb = cv.resize(square_rgb, target_size, interpolation=cv.INTER_AREA)

    # Convert to PIL Image for torchvision transforms
    pil_img = Image.fromarray(square_rgb)

    # Apply validation transforms
    val_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    tensor_img = val_transforms(pil_img)
    return tensor_img


class ChessPieceClassifier:
    def __init__(self, model_path, class_names, device='cuda'):
        """
        Initialize the classifier with a trained model.

        Args:
            model_path: Path to saved model weights (.pth file)
            class_names: List of class names in the same order as training
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names

        # Load model
        num_classes = len(class_names)
        self.model = ChessPieceCNN(num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        print(f"Model loaded successfully on {self.device}")
        print(f"Classes: {self.class_names}")

    def predict(self, square_img):
        """
        Predicts the chess piece in a square image.

        Returns:
            predicted_class: Class name (e.g., 'wk', 'bp', 'empty')
            confidence: Confidence score (0-1)
        """
        # Preprocess image
        tensor_img = preprocess_square(square_img)
        tensor_img = tensor_img.unsqueeze(0).to(self.device)  # Add batch dimension

        # Inference
        with torch.no_grad():
            outputs = self.model(tensor_img)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class = self.class_names[predicted_idx.item()]
        confidence_score = confidence.item()

        return predicted_class, confidence_score

    def predict_batch(self, square_images):
        """
        Predicts multiple squares at once (more efficient).

        Args:
            square_images: List of 64 square images

        Returns:
            predictions: List of (class_name, confidence) tuples
        """
        # Preprocess all squares
        tensor_batch = torch.stack([preprocess_square(img) for img in square_images])
        tensor_batch = tensor_batch.to(self.device)

        # Batch inference
        with torch.no_grad():
            outputs = self.model(tensor_batch)
            probabilities = F.softmax(outputs, dim=1)
            confidences, predicted_indices = torch.max(probabilities, 1)

        # Convert to class names
        predictions = [
            (self.class_names[idx.item()], conf.item())
            for idx, conf in zip(predicted_indices, confidences)
        ]

        return predictions


def class_to_fen_char(class_name):
    """
    Converts class name to FEN character.

    FEN notation:
    - Uppercase: White pieces (K, Q, R, B, N, P)
    - Lowercase: Black pieces (k, q, r, b, n, p)
    - '1-8': Empty squares

    """
    fen_mapping = {
        'wk': 'K', 'wq': 'Q', 'wr': 'R', 'wb': 'B', 'wn': 'N', 'wp': 'P',
        'bk': 'k', 'bq': 'q', 'br': 'r', 'bb': 'b', 'bn': 'n', 'bp': 'p',
        'empty': None,
        '': None  # In case of empty string
    }

    return fen_mapping.get(class_name.lower(), None)


def generate_fen_from_predictions(predictions, board_orientation='white'):
    """
    Generates FEN string from board predictions.

    Args:
        predictions: List of 64 (class_name, confidence) tuples in row-major order
        board_orientation: 'white' (bottom) or 'black' (bottom)

    Returns:
        fen_string: FEN position string (piece placement only)
    """
    # Extract class names
    class_names = [pred[0] for pred in predictions]

    # If board is from black's perspective, reverse the order
    if board_orientation == 'black':
        class_names = class_names[::-1]

    # Build FEN string row by row (rank 8 to rank 1)
    fen_rows = []

    for rank in range(8):
        row_start = rank * 8
        row_end = row_start + 8
        row_classes = class_names[row_start:row_end]

        # Convert to FEN characters
        fen_chars = [class_to_fen_char(cls) for cls in row_classes]

        # Compress empty squares
        fen_row = ""
        empty_count = 0

        for char in fen_chars:
            if char is None:
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += char

        # Add remaining empty squares
        if empty_count > 0:
            fen_row += str(empty_count)

        fen_rows.append(fen_row)

    # Join rows with '/'
    fen_string = '/'.join(fen_rows)

    return fen_string


def detect_board_orientation(predictions):
    """
    Robust orientation detection using multiple heuristics.
    Returns: 'white' or 'black'
    """
    board = [pred[0] for pred in predictions]

    scores = {'white': 0, 'black': 0}

    # Heuristic 1: King positions
    for i, piece in enumerate(board):
        row = i // 8

        if piece == 'wk':
            if row >= 6:  # White king in bottom rows
                scores['white'] += 5
            elif row <= 1:  # White king in top rows
                scores['black'] += 5

        elif piece == 'bk':
            if row >= 6:  # Black king in bottom rows
                scores['black'] += 5
            elif row <= 1:  # Black king in top rows
                scores['white'] += 5

    # Heuristic 2: Pawn positions (pawns move forward)
    for i, piece in enumerate(board):
        row = i // 8

        if piece == 'wp':
            if row >= 5:  # White pawns near bottom (haven't moved far)
                scores['white'] += 2
            elif row <= 2:  # White pawns near top (moved forward)
                scores['black'] += 2

        elif piece == 'bp':
            if row >= 5:  # Black pawns near bottom (moved forward)
                scores['white'] += 2
            elif row <= 2:  # Black pawns near top (haven't moved far)
                scores['black'] += 2

    # Heuristic 3: Overall piece density
    bottom_half_white = sum(1 for i in range(32, 64) if board[i].startswith('w'))
    bottom_half_black = sum(1 for i in range(32, 64) if board[i].startswith('b'))
    top_half_white = sum(1 for i in range(0, 32) if board[i].startswith('w'))
    top_half_black = sum(1 for i in range(0, 32) if board[i].startswith('b'))

    if bottom_half_white > top_half_white:
        scores['white'] += 1
    if bottom_half_black > top_half_black:
        scores['black'] += 1

    # Return orientation with highest score
    if scores['white'] > scores['black']:
        return 'white'
    elif scores['black'] > scores['white']:
        return 'black'
    else:
        # Default fallback
        print("Warning: Could not confidently determine orientation, defaulting to 'white'")
        return 'white'


class ChessMateApp:
    def __init__(self, model_path, class_names, device='cuda'):
        """
        Initialize the CheckMate Machine application.
        """
        self.classifier = ChessPieceClassifier(model_path, class_names, device)
        print("CheckMate Machine initialized successfully!")

    def capture_screenshot(self, region=None):
        """
        Captures a screenshot of the screen.

        Args:
            region: Tuple (x, y, width, height) for specific region, or None for full screen

        Returns:
            numpy array (BGR format for OpenCV)
        """
        screenshot = pyautogui.screenshot(region=region)
        # Convert PIL to OpenCV format (RGB -> BGR)
        screenshot_np = np.array(screenshot)
        screenshot_bgr = cv.cvtColor(screenshot_np, cv.COLOR_RGB2BGR)
        return screenshot_bgr

    def process_board_image(self, img, auto_detect_orientation, save_debug=False):
        """
        Complete pipeline: detect board, extract squares, classify, generate FEN.

        Args:
            img: Input image (BGR format)
            board_orientation: 'white' or 'black' (which side is at bottom)
            save_debug: Whether to save debug images

        Returns:
            fen_string: FEN notation of the board
            predictions: List of (class, confidence) for each square
            warped_board: Extracted board image
        """
        print("Processing board image...")

        corners = find_chessboard_corners(img)
        if corners is None:
            raise ValueError("Could not detect chessboard in image!")
        print("Chessboard detected")
        warped_board = extract_chessboard(img, corners)
        print("Board extracted and warped")
        squares = split_board_into_squares(warped_board)
        print("Board split into 64 squares")
        predictions = self.classifier.predict_batch(squares)
        print("All pieces classified")

        if auto_detect_orientation:
            board_orientation = detect_board_orientation(predictions)
            print(f"Detected orientation: {board_orientation.upper()} at bottom")
        else:
            board_orientation = 'white'

        fen_string = generate_fen_from_predictions(predictions, board_orientation)
        print(f"FEN generated: {fen_string}")

        return fen_string, predictions, warped_board

    def process_screenshot(self, region=None, auto_detect_orientation=True):
        """
        Capture screenshot and process it.
        """
        print("\nCapturing screenshot...")
        img = self.capture_screenshot(region)
        return self.process_board_image(img, auto_detect_orientation=auto_detect_orientation)

    def process_image_file(self, image_path, auto_detect_orientation=True):
        """
        Load image from file and process it.
        """
        print(f"\nLoading image from {image_path}...")
        img = cv.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        return self.process_board_image(img, auto_detect_orientation=auto_detect_orientation)

    def continuous_monitoring(self, region=None, interval=3, auto_detect_orientation=True):
        """
        Continuously monitor and process board at regular intervals.

        Args:
            region: Screen region to capture
            interval: Seconds between captures
        """
        print(f"\nStarting continuous monitoring (every {interval} seconds)")
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                try:
                    fen_string, predictions, _, orientation = self.process_screenshot(
                        region,
                        auto_detect_orientation
                    )

                    print(f"\n{'=' * 60}")
                    print(f"Orientation: {orientation.upper()} pieces at bottom")
                    print(f"FEN: {fen_string}")
                    print(f"{'=' * 60}\n")

                    # Display board in text format
                    self.display_board_text(predictions)

                except Exception as e:
                    print(f"Error processing board: {e}")

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\nMonitoring stopped")

    def display_board_text(self, predictions):
        """
        Display the board in text format with detected pieces.
        """
        print("\nDetected Board:")
        print("  a  b  c  d  e  f  g  h")

        for rank in range(8):
            row_start = rank * 8
            row_end = row_start + 8
            row_preds = predictions[row_start:row_end]

            # Create row string
            row_str = f"{8 - rank} "
            for class_name, confidence in row_preds:
                fen_char = class_to_fen_char(class_name)
                if fen_char is None:
                    row_str += " . "
                else:
                    row_str += f" {fen_char} "
            row_str += f" {8 - rank}"

            print(row_str)

        print("  a  b  c  d  e  f  g  h\n")


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "chess_piece_cnn.pth"  # Path to your trained model

    # Class names match training order
    CLASS_NAMES = ['bb', 'bk', 'bn', 'bp', 'bq', 'br',
                   'e',  # empty square class
                   'wb', 'wk', 'wn', 'wp', 'wq', 'wr']

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize app
    app = ChessMateApp(MODEL_PATH, CLASS_NAMES, DEVICE)

    app.continuous_monitoring(
        region=None,
        interval=3,   # seconds between captures
        auto_detect_orientation=True
    )

    print("\nProcessing complete!")
import torch
import torchvision.transforms as transforms
from PIL import Image

# ==========================================
# 1. Define the Reusable CNN Transformation
# ==========================================
# Adjust IMAGE_SIZE, MEAN, and STD if your model used custom values during training.
IMAGE_SIZE = 224  # Standard for many CNNs (e.g., ResNet). Change to 64, 128, etc., if needed.
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

inference_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
])

# ==========================================
# 2. Implement the 64-Square Slicing Function
# ==========================================
def slice_chessboard_to_tensors(image_path: str, transform_pipeline: transforms.Compose) -> torch.Tensor:
    """
    Slices a standard chessboard image into 64 uniform squares and prepares 
    them as a single batched tensor for CNN inference.
    
    Args:
        image_path (str): Path to the cropped, top-down chessboard image.
        transform_pipeline (transforms.Compose): Torchvision transform pipeline.
        
    Returns:
        torch.Tensor: A batch tensor of shape (64, C, H, W) ready for the CNN.
    """
    # Open image and ensure it's in RGB mode
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    
    # Calculate uniform step sizes for an 8x8 grid
    square_w = width / 8
    square_h = height / 8
    
    square_tensors = []
    
    # Iterate through ranks (rows) and files (columns)
    # Chess arrays conventionally go from top-left (A8/H8 depending on orientation) to bottom-right
    for row in range(8):
        for col in range(8):
            # Define bounding box coordinates
            left = col * square_w
            top = row * square_h
            right = (col + 1) * square_w
            bottom = (row + 1) * square_h
            
            # Crop the individual square
            square_img = img.crop((left, top, right, bottom))
            
            # Apply torchvision pipeline (Resize -> Tensor -> Normalize)
            transformed_square = transform_pipeline(square_img)
            
            square_tensors.append(transformed_square)
            
    # Stack all 64 square tensors into a single batched tensor
    # Output shape: [64, 3, IMAGE_SIZE, IMAGE_SIZE]
    batch_tensor = torch.stack(square_tensors)
    return batch_tensor

# ==========================================
# 3. Example Usage for Model Inference
# ==========================================
if __name__ == "__main__":
    # Preprocess the entire board into a single batch
    chessboard_image_path = "detected_chessboard.jpg"
    
    try:
        input_batch = slice_chessboard_to_tensors(chessboard_image_path, inference_transforms)
        print(f"Successfully processed batch tensor shape: {input_batch.shape}")
        
        # Example Inference Step:
        # model = YourChessCNN()
        # model.load_state_dict(torch.load('chess_piece_cnn.pth'))
        # model.eval()
        # with torch.no_grad():
        #     predictions = model(input_batch) # Shape: [64, num_classes]
            
    except FileNotFoundError:
        print(f"Please provide a valid image path to replace '{chessboard_image_path}'.")
    
