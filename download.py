import logging
import socket
import urllib2
from multiprocessing import Pool
from boto.s3.connection import S3Connection
from boto.s3.connection import OrdinaryCallingFormat
from boto.s3.connection import S3ResponseError
from boto.s3.key import Key
import time
import os
import json
socket.setdefaulttimeout(30)

conf = json.loads(open("s3.config").read())
conn = S3Connection(aws_access_key_id=conf['access_key'],\
                    aws_secret_access_key=conf['secret_key'],\
                    host=conf['host'],\
                    port=int(conf['port']),\
                    is_secure=False,\
                    calling_format=OrdinaryCallingFormat(),)
bucket_root = 'me.ele.bdi.license.image'
store_dir = './licenses/'

image_list = open('training_licenses.txt','r').read().split('\n')
image_list = image_list[:15000]
def download(image_list):
    for image in image_list: 
       #bucket = bucket_root + '.' + image[:2] 
       bucket = conn.get_bucket(bucket_root)
       if bucket is not None:
            key = bucket.get_key(image)
            if key is not None:
                img = key.get_contents_to_filename(store_dir+image) 
                print 'download %s'% image

download(image_list)
