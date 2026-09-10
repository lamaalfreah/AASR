"""CPU BiGRU intent classification plus inference-only deterministic slots.

No model training or provider calls at import. Frozen Rule Baseline 0 is not used
as an inference fallback; its validator is the shared structured-query contract.
"""
from __future__ import annotations
from dataclasses import dataclass
import re
import unicodedata
from spatial.schema import CATEGORIES,DIRECTIONS,OPERATIONS,EntityReference,StructuredQuery,QueryError
from spatial.query_parser import validate_query

SEED=240431
MAX_TOKENS=96


def normalize_words(text):
    text=''.join(c for c in unicodedata.normalize('NFC',text) if unicodedata.category(c)!='Mn' and c!='ـ')
    return text.translate(str.maketrans('أإآى','اااي')).lower()


def tokenize(question):
    # Mask quoted proper names, not task cues. No context or metadata is required.
    text=re.sub(r'«.*?»',' ENTITY ',question,flags=re.S)
    text=re.sub(r'\d+(?:\.\d+)?',' NUMBER ',text)
    return re.findall(r'[\w]+|[^\w\s]',normalize_words(text))[:MAX_TOKENS]


def encode(question,vocabulary):
    return [vocabulary.get(w,1) for w in tokenize(question)] or [1]


def make_model(vocabulary_size,embedding_dim=48,hidden_dim=48):
    import torch
    from torch import nn
    class IntentBiGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding=nn.Embedding(vocabulary_size,embedding_dim,padding_idx=0)
            self.gru=nn.GRU(embedding_dim,hidden_dim,batch_first=True,bidirectional=True)
            self.dropout=nn.Dropout(.1)
            self.output=nn.Linear(hidden_dim*2,len(OPERATIONS))
        def forward(self,tokens,lengths):
            packed=nn.utils.rnn.pack_padded_sequence(self.embedding(tokens),lengths.cpu(),batch_first=True,enforce_sorted=False)
            _,hidden=self.gru(packed)
            return self.output(self.dropout(torch.cat((hidden[-2],hidden[-1]),dim=1)))
    return IntentBiGRU()


PLURALS={
    'مستشفى':('مستشفيات',),'عيادة':('عيادات',),'صيدلية':('صيدليات',),'مدرسة':('مدارس',),
    'جامعة':('جامعات',),'كلية':('كليات',),'مطعم':('مطاعم',),'مقهى':('مقاهي','كوفي'),
    'بنك':('بنوك',),'محطة وقود':('محطات الوقود','محطات وقود'),
    'مطعم وجبات سريعة':('مطاعم الوجبات السريعة',),'صراف آلي':('صرافات آلية','صرافات الآلية'),
    'روضة أطفال':('رياض الأطفال',),'مكان عبادة':('أماكن العبادة',)}


def category_mentions(question):
    # Longest overlapping label wins (restaurant vs fast food); repeated mentions
    # remain ordered. Proper names in quotes must not become category arguments.
    def mask_quote(m):
        return m[0] if m[1] in CATEGORIES else ' '*len(m[0])
    text=normalize_words(re.sub(r'«(.*?)»',mask_quote,question,flags=re.S))
    spans=[]
    for category in CATEGORIES:
        for alias in (category,)+PLURALS.get(category,()):
            pattern=r'(?<!\w)(?:و|ب|ل|ك)?(?:ال)?'+re.escape(normalize_words(alias))+r'(?!\w)'
            for m in re.finditer(pattern,text):spans.append((m.start(),m.end(),category))
    kept=[]
    for span in sorted(spans,key=lambda s:(-(s[1]-s[0]),s[0])):
        if not any(span[0]<s[1] and s[0]<span[1] for s in kept):kept.append(span)
    return sorted(kept)


def radius_from_question(question):
    text=normalize_words(re.sub(r'«.*?»',' ',question,flags=re.S))
    values=[]
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*(?:كم|كيلومتر(?:ات|ين)?)\b',text):values.append(float(m[1]))
    word_values={'واحد':1,'واحدة':1,'اثنين':2,'اثنان':2,'ثلاثة':3,'ثلاث':3,
                 'اربعة':4,'اربع':4,'خمسة':5,'خمس':5,'ثمانية':8,'ثمان':8,'عشرة':10,'عشر':10}
    for word,value in word_values.items():
        if re.search(r'(?<!\w)'+word+r'\s+كيلومتر(?:ات)?\b',text):values.append(float(value))
    if re.search(r'كيلومتر\s+ونصف',text):values.append(1.5)
    elif re.search(r'كيلومتر\s+واحد',text):values.append(1.0)
    if 'كيلومترين' in text:values.append(2.0)
    if len(set(values))!=1:raise QueryError('invalid_query','missing_or_conflicting_radius','slots')
    return values[0]


def extract_slots(operation,question,context):
    """Generic lexical slots selected by the NEURAL intent; no gold/template lookup."""
    if not context.anchor or context.errors:raise QueryError('invalid_query','invalid_context','slots')
    quoted=re.findall(r'«(.*?)»',question,flags=re.S)
    visible={p.name for p in context.candidates}|{context.anchor.name}
    # Quotes can also wrap a category; only visible names are references.
    category_quotes={m[1] for m in re.finditer(r'نوع\s+«(.*?)»',question,flags=re.S)}
    names=[n for n in quoted if n in visible and n not in category_quotes]
    if not names:
        names=[name for name in visible if name and name in question]
        names.sort(key=lambda name:question.find(name))
    anchor=context.anchor.name
    if operation=='count_within_radius':
        unique=list(dict.fromkeys(names))
        if len(unique)!=1:raise QueryError('invalid_query','count_origin_not_explicit_unique','slots')
        origin=EntityReference(unique[0],'unspecified' if unique[0]==anchor else 'candidate')
    else:
        if anchor not in names and not any(cue in question for cue in ('هذا الموقع','المكان المرجعي','هذا المكان')):
            raise QueryError('invalid_query','origin_not_explicit','slots')
        origin=EntityReference(anchor,'context_anchor')
    targets=()
    if operation in ('cardinal_direction','closer_of_two'):
        remaining=list(names)
        if anchor in remaining:remaining.remove(anchor)
        expected=1 if operation=='cardinal_direction' else 2
        # Repeated mentions of the same alternatives are linguistic repetition,
        # not extra arguments; candidate multiplicity is handled by the engine.
        if len(remaining)>expected:remaining=list(dict.fromkeys(remaining))
        if len(remaining)!=expected:raise QueryError('invalid_query','wrong_named_target_count','slots')
        targets=tuple(EntityReference(name,'unspecified') for name in remaining)
    mentions=category_mentions(question)
    categories=[]
    for _,_,category in mentions:
        if category not in categories:categories.append(category)
    needs_one=operation in ('nearest_category','count_within_radius','within_radius_yes_no','spatial_multi_constraint')
    needs_two=operation in ('nearest_of_two_categories','two_hop_nearest')
    if needs_one and len(categories)!=1:raise QueryError('invalid_query','category_slot_count','slots')
    if needs_two and len(categories)!=2:raise QueryError('invalid_query','two_category_slot_count','slots')
    first=second=None
    if operation=='two_hop_nearest':
        first,second=categories
        # Nested nearest phrasing expresses the final category before its reference:
        # "أقرب C2 إلى/من أقرب C1". Sequential "أولا ... ثم" keeps surface order.
        if not re.search(r'اولا|اول(?:\s|،)|خطوتين|مرحلتين|ثم|بعدين|بعدها',normalize_words(question)):
            if re.search(r'اقرب\s+.+?\s+(?:الي|من|لا?قرب)\s*(?:موقع\s+)?اقرب',normalize_words(question)):
                first,second=second,first
        categories=[]
    elif not(needs_one or needs_two):categories=[]
    radius=radius_from_question(question) if operation in ('count_within_radius','within_radius_yes_no','spatial_multi_constraint') else None
    direction=None
    if operation=='spatial_multi_constraint':
        unquoted=re.sub(r'«.*?»',' ',question,flags=re.S)
        directions=[d for d in DIRECTIONS if re.search(r'(?<!\w)(?:ب|ل)?(?:ال)?'+d+r'(?!\w)',unquoted)]
        if len(directions)!=1:raise QueryError('invalid_query','direction_slot_count','slots')
        direction=directions[0]
    query=StructuredQuery(operation,origin,targets,tuple(categories),radius,direction,first,second)
    validate_query(query)
    return query


@dataclass
class ShortPrediction:
    operation: str
    confidence: float
    query: StructuredQuery | None
    status: str
    reason: str | None = None


class NeuralShortParser:
    def __init__(self,artifact_dir):
        import json,torch
        from pathlib import Path
        self.torch=torch
        config=json.loads((Path(artifact_dir)/'model_config.json').read_text())
        self.vocabulary=config['vocabulary']
        self.model=make_model(len(self.vocabulary))
        self.model.load_state_dict(torch.load(Path(artifact_dir)/'intent_bigru.pt',map_location='cpu',weights_only=True))
        self.model.eval()
    def predict(self,question,context):
        torch=self.torch
        ids=encode(question,self.vocabulary)
        with torch.inference_mode():
            probs=self.model(torch.tensor([ids],dtype=torch.long),torch.tensor([len(ids)])).softmax(-1)[0]
        idx=int(probs.argmax());op=OPERATIONS[idx];confidence=float(probs[idx])
        try:return ShortPrediction(op,confidence,extract_slots(op,question,context),'success')
        except QueryError as exc:return ShortPrediction(op,confidence,None,exc.status,exc.reason)
