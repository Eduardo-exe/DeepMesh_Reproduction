import numpy as np

class MeshDecoder:
    """
    Decodificador responsável por converter os tokens inferidos pelo Transformer
    em coordenadas espaciais 3D contínuas.
    Inspirado no bloco `deserialize` de data_utils do DeepMesh.
    """
    def __init__(self, patch_size=4, block_size=8, offset_size=16):
        self.patch_size = patch_size
        self.block_size = block_size
        self.offset_size = offset_size
        
    def undiscretize(self, t, continuous_range=(-1, 1), num_discrete=512):
        lo, hi = continuous_range
        t = t.astype(np.float32)
        t += 0.5
        t /= num_discrete
        return t * (hi - lo) + lo

    def deserialize(self, sequence):
        """
        Converte uma sequência de IDs (vocab_size) em vértices no espaço 3D real.
        """
        vertices = []
        for token in sequence:
            if token > 0:
                # Simulação da conversão token -> coordenadas discretas
                x = (token % 512)
                y = ((token // 512) % 512)
                z = ((token // 262144) % 512)
                vertices.append([x, y, z])
                
        if len(vertices) == 0:
            print("[Aviso] Sequência de tokens vazia. Gerando Fallback.")
            vertices = np.random.rand(100, 3) * 512
        else:
            vertices = np.array(vertices)
            
        # Converter para o espaço -1 a 1
        vertices = self.undiscretize(vertices)
        
        return vertices
