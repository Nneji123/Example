import os 

if os.path.exists("toxicx_model_0.pth"):
    print("Model Exists")
else:
    os.system("wget https://huggingface.co/spaces/EuroPython2022/ToxicCommentClassification/resolve/main/toxicx_model_0.pth")
    print("Model Downloaded!")