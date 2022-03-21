import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from text_simlirity import Text_Similarity
from feature_extract.feature_extract import extract_feature
from model.rerank_xgb import predict
from sanic import Sanic, Blueprint, response

import traceback

ts = Text_Similarity()

app = Blueprint(__name__)

@app.route('/score', methods=['POST'])
def search(request):
    try:
        data = request.json
        query = data['query']
        title = data['title']
        doc = data['doc']
        doc = title+'。'+doc
        features = extract_feature(query, doc)
        total_score = predict([features])[0]
        return response.json({
                **data,
                "total_score": float(total_score)
            }, ensure_ascii=False)
    except Exception as e:
        exc_info = 'Exception: {}\n{}'.format(e, traceback.format_exc())
        return response.json({'error': exc_info}, ensure_ascii=False)


server = Sanic(__name__)
server.blueprint(app)

if __name__ == '__main__':
    server.run(port=8001, debug=False, host='0.0.0.0')