from .segmentations import (ISIC_seg, ISICattribute_seg, Bkaiattributes_seg, busiattributes_seg, 
                            camusattributes_seg, ISIC_image, Bioparse_image, Bioparse_segmentation, 
                            Bioparse_segmentation2, Bioparse_amos22, Bioparse_segmentation_amos22, Bioparse_camus, Bioparse_amos22_weakly)


def build_dataset(config, preprocessors, inter_mode=True):
    
    if config.name == "isic_seg":
        return build_isic_dataset(config, preprocessors, inter_mode)
    elif config.name == "isic_attr":
        return build_isicattr_dataset(config, preprocessors, inter_mode)
    elif config.name == "bkai_attr":
        return build_bkaiattr_dataset(config, preprocessors, inter_mode)
    elif config.name == "busi_attr":
        return build_busiattr_dataset(config, preprocessors, inter_mode)
    elif config.name == "camus_attr":
        return build_camusattr_dataset(config, preprocessors, inter_mode)
    elif config.name == "isic_image":
        return build_isic_image_dataset(config, preprocessors)
    elif config.name == "bioparse_image":
        return build_bioparse_image_dataset(config, preprocessors)
    elif config.name == "bioparse_seg":
        return build_bioparse_seg_dataset(config, preprocessors)
    elif config.name == "bioparse_seg2":
        return build_bioparse_seg2_dataset(config, preprocessors)
    elif config.name == "bioparse_amos22":
        return build_bioparse_amos22_dataset(config, preprocessors)
    elif config.name == "bioparse_seg_amos22":
        return build_bioparse_seg_amos22_dataset(config, preprocessors)
    elif config.name == "bioparse_seg_camus":
        return build_bioparse_seg_camus_dataset(config, preprocessors)
    elif config.name == "bioparse_amos22_weakly":
        return build_bioparse_amos22_weakly_dataset(config, preprocessors)
    



def build_isic_dataset(config, preprocessors, inter_mode=True):
    if inter_mode:
        config.inter_dir = None
        config.inter_layer = None
    return ISIC_seg(preprocessors, 
                    config.prompt_type, 
                    config.images_dir, 
                    config.masks_dir, 
                    config.inter_dir,
                    config.inter_layer,
                    config.caps_file, 
                    config.override_prompt, 
                    config.inter_norm,
                    config.resize,
                    image_size=tuple(config.image_size) if config.image_size is not None else None,)

    

def build_isicattr_dataset(config, preprocessors, inter_mode=True):
    if inter_mode:
        config.inter_dir = None
        config.inter_layer = None
    return ISICattribute_seg(preprocessors, 
                             config.prompt_type,
                             config.images_dir, 
                             config.masks_dir, 
                             config.inter_dir,
                             config.inter_layer,
                             config.caps_file, 
                             image_size=tuple(config.image_size) if config.image_size is not None else None,)
    



def build_bkaiattr_dataset(config, preprocessors, inter_mode=True):
    if inter_mode:
        config.inter_dir = None
        config.inter_layer = None
    return Bkaiattributes_seg(preprocessors, 
                              config.prompt_type,
                              config.images_dir, 
                              config.masks_dir, 
                              config.inter_dir,
                              config.inter_layer,
                              config.caps_file, 
                              image_size=tuple(config.image_size) if config.image_size is not None else None,)
    

    
def build_isic_image_dataset(config, preprocessors):
    return ISIC_image(preprocessors, 
                      config.images_dir, 
                      config.masks_dir, 
                      config.layercam_dir,
                      config.caps_file, 
                      image_size=tuple(config.image_size) if config.image_size is not None else None,
                      featuremap_size=config.featuremap_size,
                      )
    


def build_busiattr_dataset(config, preprocessors, inter_mode=True):
    if inter_mode:
        config.inter_dir = None
        config.inter_layer = None
    return busiattributes_seg(preprocessors, 
                              config.prompt_type,
                              config.images_dir, 
                              config.masks_dir, 
                              config.inter_dir,
                              config.inter_layer,
                              config.caps_file, 
                              image_size=tuple(config.image_size) if config.image_size is not None else None,)
    

def build_camusattr_dataset(config, preprocessors, inter_mode=True):
    if inter_mode:
        config.inter_dir = None
        config.inter_layer = None
    return camusattributes_seg(preprocessors, 
                              config.prompt_type,
                              config.images_dir, 
                              config.masks_dir, 
                              config.inter_dir,
                              config.inter_layer,
                              config.caps_file, 
                              image_size=tuple(config.image_size) if config.image_size is not None else None,)
    


def build_isic_image_dataset(config, preprocessors):
    return ISIC_image(preprocessors, 
                      config.images_dir, 
                      config.masks_dir, 
                      config.layercam_dir,
                      config.caps_file, 
                      image_size=tuple(config.image_size) if config.image_size is not None else None,
                      featuremap_size=config.featuremap_size,
                      )


def build_bioparse_image_dataset(config, preprocessors):
    return Bioparse_image(preprocessors, 
                          config.modality, 
                          config.organ, 
                          config.root_dir, 
                          config.split, 
                          config.train_rate,
                          image_size=tuple(config.image_size) if config.image_size is not None else None,
                          featuremap_size=config.featuremap_size,
                          gcam_dir=config.gcam_dir
                          )
                          


def build_bioparse_seg_dataset(config, preprocessors):
    return Bioparse_segmentation(preprocessors, 
                                 config.modality, 
                                 config.organ, 
                                 config.root_dir, 
                                 config.split, 
                                 config.train_rate,
                                 image_size=tuple(config.image_size) if config.image_size is not None else None,
                                 resize=config.resize)


def build_bioparse_seg2_dataset(config, preprocessors):
    return Bioparse_segmentation2(preprocessors, 
                                  config.modality, 
                                  config.organ, 
                                  config.root_dir, 
                                  config.split, 
                                  config.train_rate,
                                  image_size=tuple(config.image_size) if config.image_size is not None else None,
                                  resize=config.resize,
                                  annotation_name=config.annotation_name,
                                  cbm_dir=config.cbm_dir)

    

def build_bioparse_amos22_dataset(config, preprocessors):
    return Bioparse_amos22(preprocessors, 
                           config.modality, 
                           config.organ, 
                           config.root_dir, 
                           config.split, 
                           config.train_rate,
                           image_size=tuple(config.image_size) if config.image_size is not None else None,
                           featuremap_size=config.featuremap_size,
                           gcam_dir=config.gcam_dir)




def build_bioparse_seg_amos22_dataset(config, preprocessors):
    return Bioparse_segmentation_amos22(preprocessors, 
                                        config.modality, 
                                        config.organ, 
                                        config.root_dir, 
                                        config.split, 
                                        config.train_rate,
                                        image_size=tuple(config.image_size) if config.image_size is not None else None,
                                        resize=config.resize,
                                        annotation_name=config.annotation_name,
                                        cbm_dir=config.cbm_dir)



def build_bioparse_seg_camus_dataset(config, preprocessors):
    return Bioparse_camus(preprocessors, 
                          config.view, 
                          config.organ, 
                          config.root_dir, 
                          config.split, 
                          config.train_rate,
                          image_size=tuple(config.image_size) if config.image_size is not None else None,
                          resize=config.resize,
                          annotation_name=config.annotation_name,
                          cbm_dir=config.cbm_dir)
    

def build_bioparse_amos22_weakly_dataset(config, preprocessors):
    return Bioparse_amos22_weakly(preprocessors, 
                                  config.modality, 
                                  config.organ, 
                                  config.root_dir, 
                                  config.split, 
                                  config.train_rate,
                                  image_size=tuple(config.image_size) if config.image_size is not None else None,
                                  featuremap_size=config.featuremap_size,
                                  gcam_dir=config.gcam_dir) 

