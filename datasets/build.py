from .segmentations import ISIC_seg


def build_dataset(config, preprocessors):
    
    if config.name == "isic_seg":
        return build_isic_dataset(config, preprocessors)


def build_isic_dataset(config, preprocessors):
    return ISIC_seg(preprocessors, config.prompt_type, config.images_dir, config.masks_dir, config.caps_file, config.override_prompt, config.zero_prompt)
    

