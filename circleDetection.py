import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

class Image:
    def __init__(self, path):
        self.original = self._load_image(path)
        self.name = self._extract_image_name(path)
        self.preprocessed = None
        self.circles = None
    
    def _load_image(self, path):
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Image at {path} not found or could not be loaded.")
        return image

    def _extract_image_name(self, path):
        r_edge = path.rfind(".")
        if r_edge > -1:
            path = path[:r_edge]
        l_edge = path.rfind("/" or "\\")
        if l_edge > -1:
            path = path[l_edge+1:]
        return path


class CircleDetector:
    def __init__(self, min_radius=10, max_radius=50, min_dist=20, clahe_low=10, clahe_high=50):
        # Parameters for Hough
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_dist = min_dist
        # Parameters for Preprocessing (clahe)
        self.clahe_threshold_low = clahe_low
        self.clahe_threshold_high = clahe_high

    def circleDetection(self, image_path):

        image = Image(image_path)

        self.preprocessing_image(image)

        image.circles = cv2.HoughCircles(
            image.preprocessed,                    # The processed edges image
            cv2.HOUGH_GRADIENT,             # The detection method (always use HOUGH_GRADIENT)
            dp=1,                           # Inverse accumulator resolution
            minDist=self.min_dist,          # Minimum distance between circle centers
            param1=50,                      # Higher Canny edge threshold
            param2=30,                      # Accumulator threshold for circle detection
            minRadius=self.min_radius,      # Minimum circle radius
            maxRadius=self.max_radius       # Maximum circle radius
        )

        if image.circles is not None:
            print(f"Detected {len(image.circles[0])} circles")
            self.visualize_circles(image)
            self.calculate_statistics(image)
        else:
            print("No circles detected.")


    def preprocessing_image(self, image: Image):
        gray = cv2.cvtColor(image.original, cv2.COLOR_BGR2GRAY)                         # Apply Gray filter
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))                     # Apply clahe (advanced contrast enhancement technique)
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)                                 # Apply Gaussian blur to reduce noise
        edges = cv2.Canny(blurred, self.clahe_threshold_low, self.clahe_threshold_high) # Apply Canny edge detection
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))                      # Define a small kernel for noise removal
        opened = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)                        # Apply morphological opening
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)                      # Apply morphological closing to fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))                      # Define a small kernel for dilation
        dilated = cv2.dilate(closed, kernel, iterations=1)                              # Apply dilation
        final_edges = cv2.GaussianBlur(dilated, (5, 5), 0)                              # Apply Gaussian blur to the dilated image
        final_edges = cv2.normalize(final_edges, None, 0, 255, cv2.NORM_MINMAX)         # Normalizing final image
    
        image.preprocessed = final_edges
        cv2.imwrite(f"preprocesses_image_{image.name}.png", final_edges)

    
    def visualize_circles(self, image: Image):
        # Draw the detected circles on the original image
        circles = np.uint16(np.around(image.circles))  # Round the values
        output = image.original.copy()
        for (x, y, r) in circles[0, :]:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)  # Draw the circle
            cv2.circle(output, (x, y), 2, (0, 0, 255), 3)  # Draw the center

        # Display the result
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.title("Detected Circles")
        plt.axis("off")
        plt.show()
        cv2.imwrite(f"output_{image.name}.png", output)
    

    def calculate_statistics(self, image: Image): 
        r_list = []
        center_list = []
        dist_list = []

        for (x, y, r) in image.circles[0, :]:
            r_list.append(r)
            center_list.append((float(x),float(y), float(r)))
            for (x2,y2,r2) in image.circles[0, :]:
                dist = math.hypot(float(x2) - float(x), float(y2) - float(y))
                if 0 < dist < (r+r2)*1.5:
                    dist_list.append(dist-r-r2)

        mid_r = sum(r_list) / len(r_list)
        print(f"Medium Radius: {mid_r}")
        print(f"Standard Deviation Radius: {np.std(r_list)}")

        mid_dist = sum(dist_list) / len(dist_list)
        print(f"Medium Distance: {mid_dist}")
        print(f"Standard Deviation Distance: {np.std(dist_list)}")

if __name__ == "__main__":
    #detectSmallCircles = CircleDetector(clahe_low=10, clahe_high=50, min_radius=10, max_radius=50, min_dist=20)
    #detectSmallCircles.circleDetection("img.tif")
    #detectSmallCircles.circleDetection("20240814_EcoN_10x_17BF.tif")

    detectSmallCircles = CircleDetector(clahe_low=40, clahe_high=80, min_radius=80, max_radius=150, min_dist=100)
    detectSmallCircles.circleDetection("20240814_EcoN_40x_7BF_28.tif")