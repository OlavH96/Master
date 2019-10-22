PROTOC_ZIP=protoc-3.3.0-linux-x86_64.zip
curl -OL https://github.com/google/protobuf/releases/download/v3.3.0/$PROTOC_ZIP \
 && unzip -o $PROTOC_ZIP -d ./ext bin/protoc && \
 rm -f $PROTOC_ZIP

BIN=/home/olavhus/master/Master/ext

export PYTHONPATH=$PYTHONPATH:$BIN

git clone https://github.com/tensorflow/models
cd models/research
$BIN/bin/protoc object_detection/protos/*.proto --python_out=.
python setup.py build && python setup.py install --install-lib $BIN

#PYTHONPATH=$PYTHONPATH:/usr/src/app/models/research:/usr/src/app/models/research/slim
PYTHONPATH=$PYTHONPATH:/home/olavhus/master/Master/models/research:/home/olavhus/master/Master/models/research/slim

cd ../..
