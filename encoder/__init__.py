from encoder import freq_encoder
from encoder.hashencoder import HashEncoder
from encoder.shencoder import SHEncoder
def get_encoder(encoder_type:str,encoder_configs:dict):
    """
    Construct and return the encoder Module and the output dimension of the encoding
    """
    if encoder_type == "frequency":
        multires = encoder_configs["freq_multires"]
        in_dim = encoder_configs["in_dim"]
        encode_fn, encode_dim =  freq_encoder.get_freq_embedder(multires,in_dim)
        return encode_fn, encode_dim
    elif encoder_type == "hash" or encoder_type == "hashgrid":
        in_dim = encoder_configs["in_dim"]
        num_levels = encoder_configs["hash_num_levels"]
        level_dim = encoder_configs["hash_level_dim"]
        per_level_scale = encoder_configs["hash_per_level_scale"]
        base_resolution = encoder_configs["hash_base_resolution"]
        log2_hashmap_size = encoder_configs["hash_log2_hashmap_size"]
        desired_resolution = encoder_configs["hash_desired_resolution"]

        encoder = HashEncoder(in_dim, num_levels, level_dim, per_level_scale, 
                                            base_resolution, log2_hashmap_size, desired_resolution)
                                            
        return encoder, encoder.output_dim
    elif encoder_type == "sh" or encoder_type == "sphere_harmonics": #sphere harmonics
        in_dim = encoder_configs["in_dim"]
        encoder = SHEncoder(in_dim)
        
        return encoder, encoder.output_dim
    else:
        raise NotImplementedError("Encoder type {} not implemented".format(encoder_type))

if __name__ == '__main__':
    get_encoder("hash")