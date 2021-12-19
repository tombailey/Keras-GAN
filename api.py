import tempfile

from os.path import join

from multiprocessing import Lock

from flask import send_file, Flask
from waitress import serve

from ml.pytorch.wgan.image_wgan import ImageWgan

image_wgan = ImageWgan(
    image_shape=(4, 64, 64),
    latent_space_dimension=100,
    use_cuda=False,
    generator_saved_model='generator.model',
    discriminator_saved_model='discriminator.model'
)
generation_lock = Lock()
temporary_directory = tempfile.TemporaryDirectory()

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_skins():
    with generation_lock:
        image_wgan.generate(
            sample_folder=temporary_directory.name
        )
        return send_file(join(temporary_directory.name, 'generated.png'))

@app.route('/health', methods=['GET'])
def health():
    return {}, 200

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)
