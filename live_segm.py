import cv2
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torchvision.models.segmentation as segmentation
import torch.nn as nn
from torchvision.transforms.functional import resize as F_resize, to_tensor
import argparse
from unet import UNet 


def load_model(model_name, checkpoint_path, device):
    if model_name == "unet":
        model = UNet(num_classes=2, in_channels=3, base_c=64).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    elif model_name == "deeplab":
        model = segmentation.deeplabv3_resnet50(pretrained=False, aux_loss=False)
        model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError("Unknown model name. Use 'unet' or 'deeplab'.")

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Select and load a segmentation model")
    parser.add_argument("--model", type=str, choices=["unet", "deeplab"], required=True, help="The model to load (unet or deeplab)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu)")

    args = parser.parse_args()
    model = load_model(args.model, args.checkpoint, args.device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        exit(1)

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    os.makedirs('videos', exist_ok=True)
    out = cv2.VideoWriter(f'videos/{args.model}.mp4', fourcc, 20.0, (frame_width, frame_height))

    print("Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_resized = F_resize(pil_img, (256, 256))
        input_tensor = to_tensor(pil_resized).unsqueeze(0).to(args.device)

        with torch.no_grad():
            if args.model == "deeplab":
                outputs = model(input_tensor)['out']  # [1,2,256,256] 
            else:
                outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)     # [1,2,256,256]
            probs_hand = probs[0,1,:,:].cpu().numpy()  # (256,256)

        mask = (probs_hand > 0.5).astype(np.uint8)
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay = frame.copy()
        overlay[mask_resized == 1] = (0,0,255)  
        alpha = 0.5
        result_frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)

        cv2.imshow("Live Segmentation", result_frame)
        out.write(result_frame)  

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
