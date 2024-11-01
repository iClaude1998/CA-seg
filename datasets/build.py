from .segmentations import ISIC_seg


def build_dataset(config, preprocessors, inter_mode=True):
    
    if config.name == "isic_seg":
        return build_isic_dataset(config, preprocessors, inter_mode)


def build_isic_dataset(config, preprocessors, inter_mode=True):
    if inter_mode:
        config.inter_dir = None
        config.inter_layer = None
    return ISIC_seg(preprocessors, 
                    config.prompt_type, 
                    config.images_dir, 
                    config.masks_dir, 
                    config.sdf_dir,
                    config.inter_dir,
                    config.inter_layer,
                    config.caps_file, 
                    config.override_prompt, 
                    config.zero_prompt,
                    image_size=tuple(config.image_size) if config.image_size is not None else None,)
    

