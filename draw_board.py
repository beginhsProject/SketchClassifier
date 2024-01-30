import tkinter as tk
import numpy as np
import cairo
from scipy.interpolate import interp1d
from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean
from rdp import rdp

class SketchApp:
    def __init__(self, predict):
        # Initialize the TKinter window with white background, width 1000 and height 600.
        self.root = tk.Tk()
        self.root.title("Sketch Drawing")
        self.canvas = tk.Canvas(self.root, width=1000, height=600, bg="white")
        self.canvas.pack()

        self.drawing_data = []  # List of the drawing data
        self.current_stroke = [] # List to store the current stroke

        self.canvas.bind("<Button-1>", self.start_stroke) # Bind the click action to creating a stroke
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke) # Bind the releasing click action to ending the stroke
        self.canvas.bind("<B1-Motion>", self.draw) # Bind mouse movement to drawing the stroke
        self.predict = predict # A function which takes in an image and predicts it
        save_button = tk.Button(self.root, text="Save Sketch", command=self.save_sketch) # Create a save sketch button
        save_button.pack()
        clear_button = tk.Button(self.root, text="Clear Canvas", command=self.clear_canvas) # Create a clear button
        clear_button.pack()
        self.text_display = tk.Label(self.root, text="") # Create a textbox to display the model's prediction
        self.text_display.pack()
    def start_stroke(self, event):
        self.current_stroke = [[], []]  # Reset current stroke
        self.add_point(event.x, event.y) # Add the starting point to the stroke
    def save_sketch(self):
      pass
    def clear_canvas(self):
      self.canvas.delete("all") # clear the canvas
      self.drawing_data = [] # reset the drawing data
    def end_stroke(self, event):
        self.add_point(event.x, event.y) # add the last point to the current stroke
        self.drawing_data.append(self.current_stroke) # save the current stroke
        predicted_text = self.predict(self.drawing_data) # predict the current drawing
        self.text_display.config(text=predicted_text) # display the prediction
        Preprocess(self.drawing_data)

    def add_point(self, x, y):
        self.current_stroke[0].append(x) 
        self.current_stroke[1].append(y)

    def draw(self, event):
        self.add_point(event.x, event.y)
        if len(self.current_stroke[0]) > 1:
            # Draw line segment connecting the last two points
            self.canvas.create_line(
                self.current_stroke[0][-2], self.current_stroke[1][-2],
                self.current_stroke[0][-1], self.current_stroke[1][-1],
                fill="black", 
            )

    def run(self):
        self.root.mainloop()

    def get_drawing_data(self):
        return self.drawing_data

def ramer_douglas_peucker(points):
    # An algorithm to simplify the drawing
    # This algorithm reduces the number of points in an approximation by a series of points of a curve (drawing in our case)
    return rdp(points,epsilon=2.0)

def vector_to_raster(vector_images, side=28, line_diameter=16, padding=16, bg_color=(0,0,0), fg_color=(1,1,1)):
    """
    An algorithm to convert from the simpilified version of the data to the numpy bitmap version
    Taken from https://github.com/googlecreativelab/quickdraw-dataset/issues/19#issuecomment-402247262
    """
    original_side = 256.    
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_width(line_diameter)

    # scale to match the new size
    # add padding at the edges for the line_diameter
    # and add additional padding to account for antialiasing
    total_padding = padding * 2. + line_diameter
    new_scale = float(side) / float(original_side + total_padding)
    ctx.scale(new_scale, new_scale)
    ctx.translate(total_padding / 2., total_padding / 2.)

    raster_images = []
    for vector_image in vector_images:
        # clear background
        ctx.set_source_rgb(*bg_color)
        ctx.paint()
        
        bbox = np.hstack(vector_image).max(axis=1)
        offset = ((original_side, original_side) - bbox) / 2.
        offset = offset.reshape(-1,1)
        centered = [stroke + offset for stroke in vector_image]

        # draw strokes, this is the most cpu-intensive part
        ctx.set_source_rgb(*fg_color)        
        for xv, yv in centered:
            ctx.move_to(xv[0], yv[0])
            for x, y in zip(xv, yv):
                ctx.line_to(x, y)
            ctx.stroke()

        data = surface.get_data()
        raster_image = np.copy(np.asarray(data)[::4])
        raster_images.append(raster_image)
    
    return raster_images
def Preprocess(drawing):
    # Create an array of all the x and y respectively
    all_x = np.concatenate([stroke[0] for stroke in drawing])
    all_y = np.concatenate([stroke[1] for stroke in drawing])

    # Align the drawing so that 0 is the top left corner
    min_x, min_y = np.min(all_x), np.min(all_y)
    aligned_drawing = [[[x - min_x for x in stroke[0]], 
                        [y - min_y for y in stroke[1]]] for stroke in drawing]

    # Scale the drawing so that it fits a 256 by 256 screen
    max_dim = max(np.max(all_x) - min_x, np.max(all_y) - min_y)
    scale_factor = 255 / max_dim
    scaled_drawing = [[[x * scale_factor for x in stroke[0]], 
                       [y * scale_factor for y in stroke[1]]] for stroke in aligned_drawing]

    # Resample strokes with 1 pixel spacing and simplify using the RDP algoirthm
    preprocessed_drawing = []
    for stroke in scaled_drawing:
        if len(stroke[0]) > 1: 
            x, y = stroke
            distances = np.sqrt(np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2)
            cum_distances = np.cumsum(distances)
            total_distance = cum_distances[-1]

            # Number of points to interpolate
            num_points = int(total_distance) + 1
            if num_points < 2:
                continue

            # Interpolate points
            fx, fy = interp1d(cum_distances, x), interp1d(cum_distances, y)
            alpha = np.linspace(0, total_distance, num_points)
            rx, ry = fx(alpha), fy(alpha)
            resampled = np.stack((rx, ry), axis=-1)

            preprocessed_drawing.append(resampled)
    preprocessed_drawing = ramer_douglas_peucker(preprocessed_drawing)
    preprocessed_drawing = [[preprocessed_drawing[:, 0], preprocessed_drawing[:, 1]]]
    preprocessed_drawing = vector_to_raster([preprocessed_drawing])[0]
    # Transform the simplified drawing into the numpy bitmap format   
    preprocessed_drawing = vector_to_raster([preprocessed_drawing])[0]
    # Reshape and normalize
    preprocessed_drawing = preprocessed_drawing.reshape(1,28,28,1) / 255
    return preprocessed_drawing


if __name__ == "__main__":
    # dummy function for testing
    def predict(image_data):
      return "test"
    app = SketchApp(predict)
    app.run()
