from header_imports import *
from super_resolution_model_training import *

if __name__ == "__main__":
    
    # Begin analysis
    if len(sys.argv) != 1:

        # Build the model
        if sys.argv[1] == "model_building":
            super_resolution__analysis_obj = super_resolution_building(model_type = sys.argv[2], image_type = sys.argv[3], category = sys.argv[4])

        # Classify the images
        if sys.argv[1] == "model_training":
            super_resolution_analysis_obj = super_resolution_training(model_type = sys.argv[2], image_type = sys.argv[3], category = sys.argv[4])
        
        # Indentify and image with model
        if sys.argv[1] == "model_compare":
            pass

