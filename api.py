import tempfile

from os.path import join

from multiprocessing import Lock

from flask import send_file, Flask
from waitress import serve

from ml.pytorch.diffusion.image_diffusion import ImageDiffusion

model = ImageDiffusion(
    saved_output_folder=join('output', 'unet')
)
generation_lock = Lock()
temporary_directory = tempfile.TemporaryDirectory()

app = Flask(__name__)


@app.route('/generate', methods=['POST'])
def generate_skins():
    with generation_lock:
        temp = tempfile.NamedTemporaryFile(suffix='.png')
        model.generate()[0].save(temp)
        temp.seek(0, 0)
        return send_file(temp, download_name='generated.png')


@app.route('/health', methods=['GET'])
def health():
    return {}, 200


if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)
