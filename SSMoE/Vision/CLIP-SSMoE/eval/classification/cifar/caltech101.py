import torch
from clipmoe import clipmoe
import torchvision
from templates import imagenet_templates
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

def zeroshot_classifier(model, classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]  # format with class
            texts = clipmoe.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

    
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clipmoe.loadMoE('/weka/s224363215/Projects/AAAI2026/SVDMoE/Vision/CLIP-MoE/CLIP-MoE/clip-moe-4-2-sharegpt4v.pt',top_k=2,num_experts=4,moe_layers=24, device=device)   
model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # or whatever size you need
    transforms.ToTensor()
])
data_path="Vision/CLIP-MoE/eval/classification/data/caltech-101/caltech-101/101_ObjectCategories"
testset = ImageFolder(root=data_path, transform=transform)#torchvision.datasets.Caltech101(root="/weka/s224363215/Projects/AAAI2026/SVDMoE/Vision/CLIP-MoE/eval/classification/data/caltech-101/caltech-101", download=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=8)
text_feature = zeroshot_classifier(model, testset.classes, imagenet_templates) #testset.classes

correct = 0
total = 0
with torch.no_grad():
    i = 0
    for data in testset:
        images, labels = data
        images = preprocess(to_pil_image(images)).unsqueeze(0).to(device)
        
        image_feature = model.encode_image(images)
        image_feature = image_feature/image_feature.norm(dim=-1, keepdim=True)
        
        sims = image_feature @ text_feature
        
        pred = torch.argmax(sims, dim=1).item()
        if labels == pred:
            correct += 1
        total += 1

    print(correct)
    print(total)
    print(correct/total)
        
print("Accuracy of the CLIP-MoE model on the Caltest101 test images: %d %%" % (100 * correct / total))