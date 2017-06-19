from flask import Flask, request
from flask import render_template
# from detect import predict
from cnn import predict
import base64
import time

app = Flask(__name__)

@app.route('/', methods=['GET'])
def main():
    return render_template('main.html')

@app.route('/detect', methods=['POST'])
def detect():
    buf = request.values['image']
    print len(buf)
    buf = buf[buf.find('base64') + 6:]
    buf = base64.b64decode(buf)
    print 'image received'
    if len(buf) == 0:
        print 'no image captured'
    start = time.time()
    find = predict(buf)
    print '%s seconds' % (time.time() - start)
    return 'Id card exists.' if find else 'No id card!'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
