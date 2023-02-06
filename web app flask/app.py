from flask import Flask,redirect,url_for,render_template,request
from model import extract_plac_text

app = Flask(__name__)


import os
BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')

@app.route('/', methods=['POST','GET']) 
def index():
    if request.method == 'POST':
        upload_file = request.files['image_name']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH,filename)
        upload_file.save(path_save)
        plac,text = extract_plac_text(path_save,filename)

        return render_template('index.html',upload=True,upload_image=filename,text=text)
    return render_template('index.html',upload=False)

if __name__ == '__main__':
    app.run(debug=True)

