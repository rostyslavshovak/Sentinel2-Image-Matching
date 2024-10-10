#Function to extract features from images using U-Net
def extract_features(image_path, model):
    image_tensor = load_and_preprocess_image_for_model(image_path)
    with torch.no_grad():
        #Extract features from the bottleneck layer of U-Net
        features = model(image_tensor)
    return features.squeeze(0).cpu().numpy()
