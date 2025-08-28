import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

def overlay_all_images(folder_path, output_path=None, target_size=(224, 224), blend_mode='average'):
    """
    Overlay all images in a folder into one single image.
    
    Args:
        folder_path: Path to folder containing images
        output_path: Path to save overlayed image (optional)
        target_size: Size to resize all images to (width, height)
        blend_mode: 'average', 'multiply', 'screen', or 'max'
    
    Returns:
        PIL Image object of the overlayed image
    """
    folder_path = Path(folder_path)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    image_files = [f for f in folder_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print("No image files found!")
        return None
    
    print(f"Found {len(image_files)} images")
    
    # Load and resize images
    images = []
    for img_path in sorted(image_files):
        try:
            img = Image.open(img_path)
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Resize to target size
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            images.append(np.array(img))
            print(f"Loaded: {img_path.name}")
        except Exception as e:
            print(f"Error loading {img_path.name}: {e}")
    
    if not images:
        print("No valid images loaded!")
        return None
    
    # Stack all images
    image_stack = np.stack(images, axis=0)  # Shape: (num_images, height, width, 3)
    
    # Apply blending mode
    if blend_mode == 'average':
        overlayed = np.mean(image_stack, axis=0)
    elif blend_mode == 'multiply':
        overlayed = np.prod(image_stack / 255.0, axis=0) * 255
    elif blend_mode == 'screen':
        # Screen blend: 1 - (1-a) * (1-b)
        normalized = image_stack / 255.0
        result = 1.0
        for img in normalized:
            result = result * (1.0 - img) 
        overlayed = (1.0 - result) * 255
    elif blend_mode == 'max':
        overlayed = np.max(image_stack, axis=0)
    else:
        overlayed = np.mean(image_stack, axis=0)
    
    # Convert back to PIL Image
    overlayed = np.clip(overlayed, 0, 255).astype(np.uint8)
    result_img = Image.fromarray(overlayed)
    
    # Save if output path provided
    if output_path:
        result_img.save(output_path, 'PNG', dpi=(300, 300))
        print(f"Overlayed image saved to: {output_path}")
    
    print(f"Successfully overlayed {len(images)} images using {blend_mode} blend mode")
    return result_img

def combine_with_red_overlay(fish_image, overlay_image_path, output_path=None, alpha=0.5, red_intensity=255):
    """
    Combine a fish image with another image that's converted to red overlay.
    
    Args:
        fish_image: PIL Image object of the fish (from overlay_all_images)
        overlay_image_path: Path to the image to convert to red overlay
        output_path: Path to save combined image (optional)
        alpha: Transparency of red overlay (0.0 to 1.0)
        red_intensity: Intensity of red color (0 to 255)
    
    Returns:
        PIL Image object of the combined image
    """
    try:
        # Load the overlay image
        overlay_img = Image.open(overlay_image_path)
        
        # Convert to RGB if needed
        if overlay_img.mode != 'RGB':
            overlay_img = overlay_img.convert('RGB')
        
        # Resize overlay to match fish image size
        overlay_img = overlay_img.resize(fish_image.size, Image.Resampling.LANCZOS)
        
        # Convert images to numpy arrays
        fish_array = np.array(fish_image)
        overlay_array = np.array(overlay_img)
        
        # Convert overlay to grayscale for intensity mapping
        overlay_gray = np.mean(overlay_array, axis=2)
        
        # Create red overlay based on grayscale intensity
        red_overlay = np.zeros_like(fish_array)
        red_overlay[:, :, 0] = (overlay_gray / 255.0) * red_intensity  # Red channel
        red_overlay[:, :, 1] = 0  # Green channel (empty)
        red_overlay[:, :, 2] = 0  # Blue channel (empty)
        
        # Blend the images
        combined = fish_array * (1 - alpha) + red_overlay * alpha
        combined = np.clip(combined, 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        result_img = Image.fromarray(combined)
        
        # Save if output path provided
        if output_path:
            result_img.save(output_path, 'PNG', dpi=(300, 300))
            print(f"Combined image saved to: {output_path}")
        
        print(f"Successfully combined fish image with red overlay (alpha={alpha})")
        return result_img
        
    except Exception as e:
        print(f"Error combining images: {e}")
        return None

# Updated usage example:
if __name__ == "__main__":
    folder_path = "/home/julian/Documents/Spezialisierung/SPEZ/data/images"
    overlay_path = "/home/julian/Documents/Spezialisierung/SPEZ/data/overlayed_images.png"
    red_overlay_image = "../data/oreochromis niloticus_modified2.png"  # The other image to make red
    final_output = "/home/julian/Documents/Spezialisierung/SPEZ/data/final_combined.png"
    
    # First, create the overlayed fish image
    fish_overlay = overlay_all_images(folder_path, overlay_path, 
                                    target_size=(224, 224), 
                                    blend_mode='average')
    
    if fish_overlay:
        # Then combine it with the red overlay
        final_image = combine_with_red_overlay(fish_overlay, 
                                             red_overlay_image, 
                                             final_output,
                                             alpha=0.3,  # 30% red overlay
                                             red_intensity=200)
        
        if final_image:
            # Display both results
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            ax1.imshow(fish_overlay)
            ax1.set_title('Overlayed Fish Images')
            ax1.axis('off')
            
            ax2.imshow(final_image)
            ax2.set_title('Combined with Red Overlay')
            ax2.axis('off')
            
            plt.tight_layout()
            plt.show()