import torch
import yaml
from PIL import Image

from model import Baseline, FineTuneTeacherModel, KnowledgeDistillationModel
from utils import create_vocab_flickr30k, create_vocab_cocoro
from transformers import AutoProcessor, AutoModelForCausalLM

# IMAGE_PATH = r"C:\Users\Vlad\Downloads\test3.jpeg"

IMAGE_PATH = "../data/flickr30k/flickr30k-images/205842.jpg"


def caption_image(config):
    if config["language"] == "en":
        vocab = create_vocab_flickr30k(config["data_root_path_en"])
    elif config["language"] == "ro":
        vocab = create_vocab_cocoro(config["data_root_path_ro"])

    if config["use_knowledge_distilation"]:
        processor = AutoProcessor.from_pretrained("microsoft/git-base-coco", use_fast=True)
        teacher_model = FineTuneTeacherModel.load_from_checkpoint(config["pretrained_teacher_path"],
                    config=config,           
                    vocab=vocab,            
                    model_name="microsoft/git-base-coco",
                    processor=processor
                )
        model = KnowledgeDistillationModel.load_from_checkpoint(config["model_saved_path"], teacher_model=teacher_model, config=config, vocab=vocab, pretrained_student_path=config["pretrained_student_path"])
    elif config["teacher_finetuning"]:
        processor = AutoProcessor.from_pretrained("microsoft/git-base-coco", use_fast=True)
        model = FineTuneTeacherModel.load_from_checkpoint(config["model_saved_path"],
                    config=config,           
                    vocab=vocab,            
                    model_name="microsoft/git-base-coco",
                    processor=processor
                )
    else:
        model = Baseline(config, vocab)
        model.load_state_dict(torch.load(config["model_saved_path"])["state_dict"])

    model.eval()
    model.cuda()

    image = Image.open(IMAGE_PATH)

    caption = model.caption_image(image, config)
    print(caption)


if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    caption_image(config)
