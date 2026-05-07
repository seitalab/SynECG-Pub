from enum import Enum
from argparse import Namespace

import torch.nn as nn

class DGMName(Enum):

    VAE = "vae"
    DCGAN = "dcgan"
    WGAN = "wgan"
    DDPM = "ddpm"

class ModelFactory:

    @staticmethod
    def _prepare_vae(params: Namespace) -> nn.Module:
        from codes.models.dgms.vae import VariationalAutoEncoder
        encoder = ModelFactory.prepare_nn(params, key="encoder")
        decoder = ModelFactory.prepare_nn(params, key="decoder")
        return VariationalAutoEncoder(
            encoder=encoder,
            decoder=decoder,
            enc_out_dim=params.enc_out_dim,
            z_dim=params.z_dim,
            initial_beta=params.initial_beta,
            final_beta=params.final_beta,
            n_total=params.total_samples
        )
        # from codes.models.dgms.vae import VAE_check
        # return VAE_check(params.enc_out_dim, params.z_dim)

    @staticmethod
    def _prepare_dcgan(params: Namespace) -> nn.Module:
        from codes.models.dgms.dcgan import DCGAN
        generator = ModelFactory.prepare_nn(params, key="generator")
        discriminator = ModelFactory.prepare_nn(params, key="discriminator")
        return DCGAN(
            generator=generator,
            discriminator=discriminator,
            z_dim=params.z_dim
        )

    @staticmethod
    def _prepare_wgan_gp(params: Namespace) -> nn.Module:
        from codes.models.dgms.wgan import WGAN_GP
        generator = ModelFactory.prepare_nn(params, key="generator")
        discriminator = ModelFactory.prepare_nn(params, key="discriminator")
        return WGAN_GP(
            generator=generator,
            discriminator=discriminator,
            z_dim=params.z_dim,
            lambda_gp=params.lambda_gp
        )    

    @staticmethod
    def _prepare_ddpm(params: Namespace) -> nn.Module:

        from denoising_diffusion_pytorch import Unet1D
        from codes.models.dgms.ddpm import DDPM
        # Prepare Unet.
        model = Unet1D(
            dim = params.enc_out_dim,
            dim_mults = tuple(params.dim_mults),
            channels = params.enc_out_dim
        )
        model.z_dim = params.z_dim # temporal solution

        # Prepare DDPM.
        n_chunks = int(
            params.max_duration * params.target_freq
        ) // params.chunk_len
        ddpm = DDPM(
            model,
            seq_length = n_chunks,
            timesteps = params.timesteps,
            objective = 'pred_v'
        )
        ddpm.z_dim = params.z_dim

        # Set encoder and decoder.
        # encoder = ModelFactory.prepare_nn(params, key="encoder")
        # decoder = ModelFactory.prepare_nn(params, key="decoder")
        # ddpm.set_encoder_and_decoder(encoder, decoder)
        ddpm.set_embed_module(
            params.enc_out_dim, params.chunk_len, params.embed_type)
        ddpm.prep_loss(params.loss_type)
        return ddpm

    @staticmethod
    def prepare_model(params: Namespace) -> nn.Module:
        if params.dgm == DGMName.VAE.value:
            return ModelFactory._prepare_vae(params)
        elif params.dgm == DGMName.DCGAN.value:
            return ModelFactory._prepare_dcgan(params)
        elif params.dgm == DGMName.WGAN.value:
            return ModelFactory._prepare_wgan_gp(params)
        elif params.dgm == DGMName.DDPM.value:
            return ModelFactory._prepare_ddpm(params)
        else:
            raise ValueError(f"Unknown DGM model name: {params.dgm}")

    @staticmethod  
    def prepare_nn(params: Namespace, key: str) -> nn.Module:
        nn_arch = vars(params)[key]
        seqlen = int(params.max_duration * params.target_freq)
        if nn_arch == "linear_enc":
            from codes.models.nn_arch.linears import LinearEncoder
            return LinearEncoder(
                input_dim=seqlen,
                output_dim=params.enc_out_dim,
                h_dim=params.h_dim if hasattr(params, "h_dim") else 256,
                add_clf=params.is_gan
            )
        elif nn_arch == "linear_chunk_enc":
            from codes.models.nn_arch.linears import LinearChunkEncoder
            return LinearChunkEncoder(
                input_dim=seqlen,
                output_dim=params.enc_out_dim,
                chunk_len=params.chunk_len if hasattr(params, "chunk_len") else 50,
                h_dim=params.h_dim if hasattr(params, "h_dim") else 256,
                add_clf=params.is_gan,
                with_sigmoid=params.dgm != "wgan"
            )
        elif nn_arch == "rnn_enc":
            from codes.models.nn_arch.rnn import RNNEncoder
            return RNNEncoder(
                seqlen=seqlen,
                enc_out_dim=params.enc_out_dim,
                h_dim=params.h_dim if hasattr(params, "h_dim") else 256,
                gru_dim=params.h_dim if hasattr(params, "gru_dim") else 256,
                chunk_len=params.chunk_len if hasattr(params, "chunk_len") else 50,
                device=params.device,
                add_clf=params.is_gan,
                with_sigmoid=params.dgm != "wgan"
            )
        elif nn_arch == "resnet18_enc":
            from codes.models.nn_arch.resnet import ResNet18
            return ResNet18(
                params, 
                add_clf=params.is_gan,
                with_sigmoid=params.dgm != "wgan"
            )
        elif nn_arch == "cnn_enc":
            from codes.models.nn_arch.cnn import CNN01
            return CNN01(
                seqlen=seqlen,
                output_dim=params.enc_out_dim,
                h_dim=params.h_dim if hasattr(params, "h_dim") else 32, 
                add_clf=params.is_gan,
                with_sigmoid=params.dgm != "wgan"
            )
        elif nn_arch == "cnn_enc02":
            from codes.models.nn_arch.cnn import CNN02
            return CNN02(
                seqlen=seqlen,
                output_dim=params.enc_out_dim,
                h_dim=params.h_dim if hasattr(params, "h_dim") else 32, 
                add_clf=params.is_gan,
                with_sigmoid=params.dgm != "wgan"
            )        
        elif nn_arch == "linear_dec":
            from codes.models.nn_arch.linears import LinearDecoder
            return LinearDecoder(
                input_dim=params.z_dim,
                output_dim=seqlen,
                h_dim=params.h_dim if hasattr(params, "h_dim") else 256
            )
        elif nn_arch == "hier_rnn_dec":
            from codes.models.nn_arch.rnn import HierarchicalRNNDecoder
            return HierarchicalRNNDecoder(
                z_dim=params.z_dim,
                seqlen=seqlen,
                chunk_len=params.chunk_len if hasattr(params, "chunk_len") else 50,
                h_dim=params.h_dim if hasattr(params, "h_dim") else 256
            )
        elif nn_arch == "rnn_dec":
            from codes.models.nn_arch.rnn import RNNDecoder
            return RNNDecoder(
                z_dim=params.z_dim,
                seqlen=seqlen,
                chunk_len=params.chunk_len if hasattr(params, "chunk_len") else 50,
                h_dim=params.h_dim if hasattr(params, "h_dim") else 64
            )
        elif nn_arch == "resnet_dec":
            # from codes.models.nn_arch.resnet import ResNet1dDecoder
            # from codes.models.nn_arch.resnet import ResNet1dDecoder02 as ResNet1dDecoder
            from codes.models.nn_arch.resnet import ResNet1dDecoder03 as ResNet1dDecoder
            return ResNet1dDecoder(
                seqlen,
                params.z_dim, 
                params.h_dim if hasattr(params, "h_dim") else 256,
                params.chunk_len if hasattr(params, "chunk_len") else 50
            )
        else:
            raise ValueError(f"Unknown neural network architecture: {nn_arch}")


def prepare_model(params: Namespace) -> nn.Module:
    return ModelFactory.prepare_model(params)
