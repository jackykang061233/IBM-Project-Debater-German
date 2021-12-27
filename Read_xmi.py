from cassis import *

with open('/Users/kangchieh/Downloads/xmi_ttlab/TypeSystem.xml', 'rb') as f:
    typesystem = load_typesystem(f)

with open('/Users/kangchieh/Downloads/xmi_ttlab/8/8835989_589455.xmi.xmi', 'rb') as f:
   cas = load_cas_from_xmi(f, typesystem=typesystem)

for sentence in cas.select('cas:Sofa'):
    print(sentence)