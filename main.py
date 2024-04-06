import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from tinyps_utils import calculate_entropy, get_avg_code_length

class TinyPhotoshop:
    def __init__(self, root):
        self.root = root
        self.setup_ui()

    def upload_image(self):
        f_types = [("Bitmap files", "*.bmp")]
        file_path = filedialog.askopenfilename(filetypes=f_types)
        if not file_path:  # If the user cancels the dialog, file_path will be an empty string
            return

        # Load the selected image using PIL
        self.original_image = Image.open(file_path)

        # Convert the image to a format that can be displayed in Tkinter
        self.original_photo_image = ImageTk.PhotoImage(self.original_image)

        # If an image is already displayed, remove it before displaying a new one
        if hasattr(self, 'original_image_label'):
            self.original_image_label.destroy()

        # Display the image in the image_frame
        self.original_image_label = ttk.Label(self.image_frame, image=self.original_photo_image)
        self.original_image_label.pack(side="left", padx=50, pady=10)

        # Adjust the size of image_frame to fit the image
        self.image_frame.configure(width=self.original_image.width, height=self.original_image.height)

    def parse_bmp_file(self):
        pass

    def exit(self):
        self.root.quit()

    def save_transformed_image(self):
        # Check if there is a transformed image to save
        if not hasattr(self, 'transformed_image'):
            messagebox.showerror("Error", "No transformed image to save")
            return

        # Open a "Save As" dialog
        file_path = filedialog.asksaveasfilename(defaultextension=".bmp", filetypes=[("Bitmap Files", "*.bmp")], initialfile="transformed_image.bmp")
        if not file_path:  # User cancelled the operation
            return

        # Save the transformed image
        self.transformed_image.save(file_path, "BMP")

    def update_original_image_display(self, image):
        # Convert the image to a format that can be displayed in Tkinter
        self.original_photo_image = ImageTk.PhotoImage(image)

        # If an image is already displayed, remove it before displaying a new one
        if hasattr(self, 'original_image_label'):
            self.original_image_label.destroy()

        # Display the image in the image_frame
        self.original_image_label = ttk.Label(self.image_frame, image=self.original_photo_image)
        self.original_image_label.pack(side="left", padx=50, pady=10)

    def update_transformed_image_display(self, transformed_image):
        # Convert the PIL image to a format that can be displayed in Tkinter
        self.transformed_photo_image = ImageTk.PhotoImage(transformed_image)

        # Display or update the transformed image label
        if hasattr(self, 'transformed_image_label'):
            self.transformed_image_label.destroy()

        self.transformed_image_label = ttk.Label(self.image_frame, image=self.transformed_photo_image)
        self.transformed_image_label.pack(side="right", padx=50, pady=10)

    def convert_to_grayscale(self, do_update_original_label = False):
        if not hasattr(self, 'original_image'):
            messagebox.showerror("Error", "Please upload an image first")
            return

        # Convert the PIL image to a NumPy array
        image_array = np.array(self.original_image)

        # Ensure the image is in RGB format
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            # Apply the luminosity method to convert to grayscale
            grayscale_array = np.dot(image_array[...,:3], [0.299, 0.587, 0.114])
        else:
            messagebox.showerror("Error", "Image is not in RGB format or already grayscale")
            return

        # The result will be a 2D array, so we need to convert it to an 8-bit grayscale image
        grayscale_image = Image.fromarray(np.uint8(grayscale_array))
        self.transformed_image = grayscale_image
        
        if do_update_original_label:
            self.update_original_image_display(self.transformed_image)
        else:
            self.update_original_image_display(self.original_image)
            self.update_transformed_image_display(self.transformed_image)
        
        return grayscale_image

    def perform_ordered_dithering(self):
        if not hasattr(self, 'original_image'):
            messagebox.showerror("Error", "Please upload an image first")
            return
        
        # Ensure the image is grayscale for dithering; convert if not
        grayscale_image = self.convert_to_grayscale(True)
        grayscale_array = np.array(grayscale_image)

        # Normalized Bayer 4x4 matrix
        bayer_matrix = np.array([[ 1,  9,  3, 11],
                                [13,  5, 15,  7],
                                [ 4, 12,  2, 10],
                                [16,  8, 14,  6]]) / 16.0

        # Scale the Bayer matrix to the size of the image
        threshold_matrix = np.tile(bayer_matrix, (grayscale_array.shape[0] // 4, grayscale_array.shape[1] // 4))

        # Apply ordered dithering
        dithered_image_array = grayscale_array > (threshold_matrix * 255)
        
        # Convert back to a PIL image
        dithered_image = Image.fromarray(np.uint8(dithered_image_array) * 255)
        self.transformed_image = dithered_image

        self.update_transformed_image_display(self.transformed_image)
        
        return dithered_image

    def perform_autolevel(self):
        if not hasattr(self, 'original_image'):
            messagebox.showerror("Error", "Please upload an image first")
            return

        image_array = np.array(self.original_image).astype('float')
        auto_level_array = np.zeros(image_array.shape)

        # processing each channel of color image independently
        if len(image_array.shape) == 3:
            for channel in range(image_array.shape[2]):
                min_val = np.min(image_array[..., channel])
                max_val = np.max(image_array[..., channel])

                # to prevent division by zero if image has only one solid colour
                if min_val == max_val:
                    auto_level_array[..., channel] = image_array[..., channel]
                else:
                    #stretch the pixel value range of the channel
                    auto_level_array[..., channel] = 255 * (image_array[..., channel] - min_val) / (max_val - min_val)
        # to take care of grayscale images
        else:
            min_val = np.min(image_array)
            max_val = np.max(image_array)
            
            if min_val == max_val:
                auto_level_array = image_array
            else:
                auto_level_array = 255 * (image_array - min_val) / (max_val - min_val)

        auto_leveled_image = Image.fromarray(auto_level_array.astype('uint8'))
        self.transformed_image = auto_leveled_image

        self.update_transformed_image_display(self.transformed_image)

    def update_metrics_display(self, entropy, avg_huffman_length):
        messagebox.showinfo("Huffman Coding Metrics", f"Huffman Coding Metrics:\nEntropy: {entropy:.2f} bits\nAverage Huffman Code Length: {avg_huffman_length:.2f} bits")

    def get_huffman_metrics(self):
        if not hasattr(self, 'original_image'):
            messagebox.showerror("Error", "Please upload an image first")
            return
        
        grayscale_image = self.convert_to_grayscale()
        grayscale_array = np.array(grayscale_image)
        entropy = calculate_entropy(grayscale_array)
        avg_code_length = get_avg_code_length(grayscale_array)
        self.update_metrics_display(entropy=entropy, avg_huffman_length=avg_code_length)

    def convert_to_negative(self):
        pass

    def setup_ui(self):
        self.root.title("Tiny Photoshop")
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg='light gray')
        self.create_banner()
        self.create_menu()
        self.create_image_display_area()

    def create_banner(self):
        self.banner = ttk.Label(self.root, text="Welcome to Tiny Photoshop! \nUpload a file and use the functionalities by clicking on the buttons below", #\nImplemented by Shubham Bhatia (301562778) for CMPT 820
                               background="silver", foreground="black", anchor="center", justify="center", padding=20, relief="raised", font=("Helvetica", 14))
        self.banner.pack(side="top", fill="x")

    def create_menu(self):
        self.menu_frame = tk.Frame(self.root, width=200)
        self.menu_frame.pack(side='left', fill='y', padx=10, pady=10)

        ttk.Label(self.menu_frame, text="Core Features").pack(padx=10, pady=5)
        self.upload_btn = ttk.Button(self.menu_frame, text="Upload", command=self.upload_image)
        self.upload_btn.pack(padx = 10, pady=10)
        self.exit_btn = ttk.Button(self.menu_frame, text="Exit", command=self.exit)
        self.exit_btn.pack(padx = 10, pady=10)
        self.grayscale_btn = ttk.Button(self.menu_frame, text="Convert Grayscale", command=self.convert_to_grayscale)
        self.grayscale_btn.pack(padx=10, pady=5, fill='x')
        self.dithering_btn = ttk.Button(self.menu_frame, text="Ordered Dithering", command=self.perform_ordered_dithering)
        self.dithering_btn.pack(padx=10, pady=5, fill='x')
        self.autolevel_btn = ttk.Button(self.menu_frame, text="Auto Level", command=self.perform_autolevel)
        self.autolevel_btn.pack(padx=10, pady=5, fill='x')
        self.huffman_btn = ttk.Button(self.menu_frame, text="Huffman Coding", command=self.get_huffman_metrics)
        self.huffman_btn.pack(padx=10, pady=5, fill='x')
        ttk.Separator(self.menu_frame, orient='horizontal').pack(fill='x', pady=10)

        ttk.Label(self.menu_frame, text="Optional Features").pack(padx=10, pady=5)
        self.saveimage_btn = ttk.Button(self.menu_frame, text="Save Transformed Image", command=self.save_transformed_image).pack(padx=10, pady=5, fill='x')
        self.negativeimage_btn = ttk.Button(self.menu_frame, text="Convert to Negative", command=self.convert_to_negative).pack(padx=10, pady=5, fill='x')

    def create_image_display_area(self):
        self.image_frame = tk.Frame(self.root, bg='white')
        self.image_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        

if __name__ == "__main__":
    root = tk.Tk()
    app = TinyPhotoshop(root)
    root.mainloop()