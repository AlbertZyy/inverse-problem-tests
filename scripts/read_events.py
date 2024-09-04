# TODO: finish this
import os
import struct
from google.protobuf.json_format import MessageToDict
from google.protobuf.message_factory import MessageFactory
from google.protobuf.descriptor_pool import DescriptorPool
from google.protobuf import descriptor_pb2

# 加载event.proto文件中的定义
descriptor_pool = DescriptorPool()
descriptor = descriptor_pb2.FileDescriptorSet()

with open('path/to/event.descriptor', 'rb') as f:
    descriptor.MergeFromString(f.read())
descriptor_pool.Add(descriptor.file[0])

# 创建消息工厂并注册事件描述符
message_factory = MessageFactory(descriptor_pool)
event = message_factory.GetPrototype(descriptor_pool.FindMessageTypeByName('tensorflow.Event'))

def read_tfevents_file(file_path):
    with open(file_path, 'rb') as f:
        while True:
            # 读取长度
            byte_len = f.read(8)
            if len(byte_len) == 0:
                break
            string_len, = struct.unpack('Q', byte_len)
            # 读取消息
            event_str = f.read(string_len)
            # 解析消息
            ev = event()
            ev.ParseFromString(event_str)
            event_dict = MessageToDict(ev)
            print(event_dict)

# 使用方法
file_path = 'path/to/your/events.out.tfevents'
read_tfevents_file(file_path)
