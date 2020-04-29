import os
import string
import sys
import xml.dom.minidom
import codecs

DELETECHARS = ''.join([string.punctuation, string.whitespace])
LENGTH = 50


def tokenize(text, length):
    tokens = {}
    token = []

    for i in range(0, len(text)):
        if text[i] not in DELETECHARS:
            token.append((i, text[i]))
        if len(token) == length:
            ngram = ''.join([x[1].lower() for x in token])
            if ngram not in tokens:
                tokens[ngram] = []
            tokens[ngram].append((token[0][0], token[-1][0]))
            token = token[1:]

    return tokens


def serialize_features(susp, src, features, outdir):
    impl = xml.dom.minidom.getDOMImplementation()
    doc = impl.createDocument(None, 'document', None)
    root = doc.documentElement
    root.setAttribute('reference', susp)
    doc.createElement('feature')

    for f in features:
        feature = doc.createElement('feature')
        feature.setAttribute('name', 'detected-plagiarism')
        feature.setAttribute('this_offset', str(f[1][0]))
        feature.setAttribute('this_length', str(f[1][1] - f[1][0]))
        feature.setAttribute('source_reference', src)
        feature.setAttribute('source_offset', str(f[0][0]))
        feature.setAttribute('source_length', str(f[0][1] - f[0][0]))
        root.appendChild(feature)

    doc.writexml(open(outdir + susp.split('.')[0] + '-'
                      + src.split('.')[0] + '.xml', 'w'),
                 encoding='utf-8')


class Baseline:
    def __init__(self, susp, src, outdir):
        self.susp = susp
        self.src = src
        self.susp_file = os.path.split(susp)[1]
        self.src_file = os.path.split(src)[1]
        self.susp_id = os.path.splitext(susp)[0]
        self.src_id = os.path.splitext(src)[0]
        self.output = self.susp_id + '-' + self.src_id + '.xml'
        self.detections = None
        self.outdir = outdir

    def process(self):
        self.preprocess()
        self.detections = self.compare()
        self.postprocess()

    def preprocess(self):
        susp_fp = codecs.open(self.susp, 'r', 'utf-8')
        self.susp_text = susp_fp.read()
        self.tokens = tokenize(self.susp_text, LENGTH)
        susp_fp.close()

        src_fp = codecs.open(self.src, 'r', 'utf-8')
        self.src_text = src_fp.read()
        src_fp.close()

    def compare(self):
        detections = []
        skipto = -1
        token = []
        for i in range(0, len(self.src_text)):
            if i > skipto:
                if self.src_text[i] not in DELETECHARS:
                    token.append((i, self.src_text[i]))
                if len(token) == LENGTH:
                    ngram = ''.join([x[1].lower() for x in token])
                    if ngram in self.tokens:
                        d = ((token[0][0], token[-1][0]),
                             (self.tokens[ngram][0][0],
                              self.tokens[ngram][0][1]))
                        for t in self.tokens[ngram]:
                            start_src = token[0][0]
                            start_susp = t[0]
                            while (start_susp < len(self.susp_text) and
                                   start_src < len(self.src_text) and
                                   self.src_text[start_src] == self.susp_text[start_susp]):
                                start_susp = start_susp + 1
                                start_src = start_src + 1
                                while (start_susp < len(self.susp_text) and
                                       self.susp_text[start_susp] in DELETECHARS):
                                    start_susp = start_susp + 1
                                while (start_src < len(self.src_text) and
                                       self.src_text[start_src] in DELETECHARS):
                                    start_src = start_src + 1
                            if (start_src - 1) - token[0][0] > d[0][1] - d[0][0]:
                                d = ((token[0][0], start_src), (t[0], start_susp))
                        detections.append(d)
                        skipto = d[0][1]
                        if skipto < len(self.src_text):
                            token = [(skipto, self.src_text[skipto])]
                        else:
                            break
                    else:
                        token = token[1:]

        print(len(detections))

        return detections

    def postprocess(self):
        serialize_features(self.susp_file, self.src_file, self.detections, self.outdir)


if __name__ == "__main__":
    if len(sys.argv) == 5:
        srcdir = sys.argv[2]
        suspdir = sys.argv[3]
        outdir = sys.argv[4]
        if outdir[-1] != "/":
            outdir += "/"
        lines = open(sys.argv[1], 'r').readlines()
        for line in lines:
            susp, src = line.split()
            baseline = Baseline(os.path.join(suspdir, susp),
                                os.path.join(srcdir, src), outdir)
            baseline.process()
    else:
        print('\n'.join(["Unexpected number of commandline arguments.",
                         "Usage: ./pan12-plagiarism-text-alignment-example.py {pairs} {src-dir} {susp-dir} {out-dir}"]))
