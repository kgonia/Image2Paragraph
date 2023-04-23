from PIL import Image
from clip_interrogator import Interrogator, Config


class ClipInterrogator:
    def __init__(self, device, captioner_base_model='blip'):
        self.device = device
        self.ci = Interrogator(Config(
            clip_model_name="ViT-L-14/openai",
            caption_model_name="blip2-2.7b",
            device=device
        ))

    def image_caption(self, image_src):
        image = Image.open(image_src).convert('RGB')
        generated_text = self.ci.interrogate(image)
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        print('\nStep1, Clip Interrogator caption:')
        print(generated_text)
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        return generated_text

    def image_negative_caption(self, image_src):
        image = Image.open(image_src).convert('RGB')
        generated_text = self.ci.interrogate_negative(image)
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        print('\nClip Interrogator negative caption:')
        print(generated_text)
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        return generated_text

    def image_caption_debug(self, image_src):
        return "A dish with salmon, in style of Helmut."