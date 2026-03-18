# Installation of libraries
!pip install easyocr imutils pytesseract ultralytics gspread -q
!apt-get install -y tesseract-ocr > /dev/null 2>&1

import os, urllib.request

model_path = "/content/models/plate_detector.pt"
os.makedirs("/content/models", exist_ok=True)

if not os.path.exists(model_path):
    print("⏳ Downloading YOLO plate model...")
    urllib.request.urlretrieve(
        "https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/raw/main/license_plate_detector.pt",
        model_path
    )

print("✅ All installed!")
# To import the  required libraries
import cv2, numpy as np, matplotlib.pyplot as plt
import re, time, imutils, pytesseract, os
import pandas as pd
from google.colab import files
import warnings
warnings.filterwarnings('ignore')

# ── Load YOLO ──
print("⏳ Loading YOLO...")
from ultralytics import YOLO
plate_model = YOLO("/content/models/plate_detector.pt")
print("✅ YOLO loaded!")

# ── Load EasyOCR ──
print("⏳ Loading EasyOCR...")
import torch, easyocr
HAS_GPU = torch.cuda.is_available()
reader = easyocr.Reader(['en'], gpu=HAS_GPU)
print(f"✅ EasyOCR ({'GPU' if HAS_GPU else 'CPU'})")

# ═══════════════════════════════════════════
#  UPLOAD SPREADSHEET
# ═══════════════════════════════════════════

PLATE_DB = set()
PLATE_DB_LIST = []
PLATE_FULL_DATA = {}

def load_plates_from_file():
    global PLATE_DB, PLATE_DB_LIST, PLATE_FULL_DATA

    print("\n📤 Upload your spreadsheet (.xlsx or .csv)")
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]

    if filename.endswith('.csv'):
        df = pd.read_csv(filename)
    else:
        df = pd.read_excel(filename)

    col_name = df.columns[0]

    PLATE_DB = set()
    PLATE_DB_LIST = []
    PLATE_FULL_DATA = {}

    for idx, row in df.iterrows():
        val = row[col_name]
        if pd.isna(val):
            continue
        cleaned = re.sub(r'[^A-Za-z0-9]', '', str(val).upper())
        if len(cleaned) >= 7:
            PLATE_DB.add(cleaned)
            PLATE_DB_LIST.append(cleaned)
            PLATE_FULL_DATA[cleaned] = row.to_dict()

    print(f"\n✅ Loaded {len(PLATE_DB)} plates from '{col_name}'")
    print(f"   Plates: {PLATE_DB_LIST}")
    return len(PLATE_DB)

def search_plate_in_sheet(plate_number):
    cleaned = re.sub(r'[^A-Za-z0-9]', '', str(plate_number).upper())
    if cleaned in PLATE_FULL_DATA:
        return True, PLATE_FULL_DATA[cleaned]
    return False, None

def refresh_plate_db():
    return load_plates_from_file()

load_plates_from_file()

print("\n🎯 Ready! Run Cell 3.")
# Car plate detector engine
from IPython.display import display, HTML

VALID_STATES = {
    'AN','AP','AR','AS','BR','CG','CH','DD','DL','GA',
    'GJ','HP','HR','JH','JK','KA','KL','LA','LD','MH',
    'ML','MN','MP','MZ','NL','OD','PB','PY','RJ','SK',
    'TN','TR','TS','UK','UP','WB'
}
TO_LETTER = {'0':'O','1':'I','2':'Z','3':'E','4':'A','5':'S','6':'G','7':'T','8':'B','9':'P'}
TO_DIGIT = {'O':'0','Q':'0','D':'0','I':'1','L':'1','l':'1','Z':'2','E':'3','A':'4','S':'5','G':'6','B':'8','P':'9','R':'9','C':'0','U':'0','J':'1','Y':'7','F':'7'}
AMBIG_DIGIT = {'T':['1','7']}
SIMILAR_DIGITS = {'5':['6'],'6':['5'],'8':['3','6'],'3':['8'],'0':['6','9'],'9':['0'],'1':['7'],'7':['1'],'4':['9']}
STATE_FIXES = {
    'HH':'MH','NH':'MH','HM':'MH','WH':'MH','NM':'MH','HN':'MH','MM':'MH','MK':'MH','IH':'MH','IM':'MH','MW':'MH','WM':'MH',
    'OL':'DL','DI':'DL','D1':'DL','OI':'DL','01':'DL','TH':'TN','IN':'TN','7N':'TN','TM':'TN','1N':'TN',
    'GI':'GJ','CI':'CG','C6':'CG','G1':'GJ','KH':'KA','XA':'KA','K4':'KA','KX':'KA','HA':'KA',
    'UF':'UP','UR':'UP','U9':'UP','UB':'UP','RI':'RJ','R1':'RJ','AF':'AP','4P':'AP','A9':'AP',
    'HK':'HR','MR':'HR','HE':'HR','HF':'HP','H9':'HP','W8':'WB','VB':'WB','K1':'KL','KI':'KL',
    'G4':'GA','6A':'GA','T5':'TS','75':'TS','J1':'JH','IK':'JK','JX':'JK','M9':'MP','8R':'BR','5K':'SK','PV':'PY','A5':'AS','7R':'TR',
    'TI':'TN','YP':'UP','4P':'AP','1S':'TS','15':'TS','T5':'TS','7S':'TS','75':'TS','YU':'UP'
}
SIMILAR = {
    'M':['H','N','W'],'H':['M','N','K'],'N':['M','H'],'K':['X','H','R'],'W':['M'],'C':['G','O'],'G':['C','6'],
    'U':['V','J'],'V':['U','Y'],'P':['R','F'],'R':['P','K'],'E':['F','B'],'F':['E','P'],'D':['O','0'],'B':['8','R'],
    'S':['5'],'A':['4','R'],'T':['7','1'],'I':['1','L'],'Z':['2','7'],'J':['1','I'],'L':['1','I'],'O':['0','Q','D'],
    'Q':['0','O'],'Y':['V','7'],'X':['K'],
}
SIMILAR_CHARS = {**SIMILAR, **SIMILAR_DIGITS}

LAST_PLATE_BBOX = None

# ═══════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════

def longest_common_substring(s1, s2):
    if not s1 or not s2:
        return "", 0
    m = len(s1); n = len(s2)
    result = ""; max_len = 0
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    result = s1[i-max_len:i]
            else:
                dp[i][j] = 0
    return result, max_len

def clean_plate_text(t):
    return re.sub(r'[^A-Za-z0-9]', '', str(t).upper())

def exact_position_compare(a, b):
    a = clean_plate_text(a)
    b = clean_plate_text(b)
    min_len = min(len(a), len(b))
    matches = 0
    diffs = 0
    diff_positions = []
    for i in range(min_len):
        if a[i] == b[i]:
            matches += 1
        else:
            diffs += 1
            diff_positions.append((i, a[i], b[i]))
    diffs += abs(len(a) - len(b))
    return matches, diffs, diff_positions, max(len(a), len(b))

def suffix_position_compare(a, b, n=6):
    a = clean_plate_text(a)[-n:]
    b = clean_plate_text(b)[-n:]
    min_len = min(len(a), len(b))
    matches = 0
    diffs = 0
    diff_positions = []
    for i in range(min_len):
        if a[i] == b[i]:
            matches += 1
        else:
            diffs += 1
            diff_positions.append((i, a[i], b[i]))
    diffs += abs(len(a) - len(b))
    return a, b, matches, diffs, diff_positions, max(len(a), len(b))

# ═══════════════════════════════════════════
# PLATE FINDING
# ═══════════════════════════════════════════

def find_plate_yolo(image, model):
    results = model(image, verbose=False)
    cands = []
    h, w = image.shape[:2]
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.3:
                continue
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            x1=max(0,x1); y1=max(0,y1); x2=min(w,x2); y2=min(h,y2)
            bw=x2-x1; bh=y2-y1
            if bw<20 or bh<10:
                continue
            px=max(15,int(bw*0.08)); py=max(10,int(bh*0.15))
            cands.append({
                'x1':max(0,x1-px),'y1':max(0,y1-py),'x2':min(w,x2+px),'y2':min(h,y2+py),
                'score':conf*20,'contour':None,'method':f'yolo({conf:.0%})','yolo_conf':conf
            })
    return cands

def find_white_plates(image):
    h,w=image.shape[:2]
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    c=[]
    for sm,vm in [(50,160),(70,140),(40,180)]:
        mask=cv2.inRange(hsv,np.array([0,0,vm]),np.array([180,sm,255]))
        k=cv2.getStructuringElement(cv2.MORPH_RECT,(25,7))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
        for cnt in imutils.grab_contours(cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)):
            x,y,bw,bh=cv2.boundingRect(cnt)
            if bh==0: continue
            ar=bw/float(bh); ap=(bw*bh)/(w*h)
            if 2<=ar<=6.5 and bw>60 and bh>15 and 0.003<ap<0.12:
                peri=cv2.arcLength(cnt,True)
                approx=cv2.approxPolyDP(cnt,0.02*peri,True)
                s=5+(3 if y>h*0.4 else 0)+(2 if 0.008<ap<0.06 else 0)+(3 if len(approx)==4 else 0)
                c.append({'x1':x,'y1':y,'x2':x+bw,'y2':y+bh,'score':s,'contour':approx if len(approx)==4 else None,'method':'white'})
    return c

def find_contour_plates(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    h,w=gray.shape
    c=[]
    for bk in [11,13]:
        bf=cv2.bilateralFilter(gray,bk,17,17)
        for lo,hi in [(30,200),(50,150),(20,250)]:
            edged=cv2.Canny(bf,lo,hi)
            edged=cv2.dilate(edged,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)),iterations=1)
            for cnt in sorted(imutils.grab_contours(cv2.findContours(edged,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)), key=cv2.contourArea, reverse=True)[:15]:
                peri=cv2.arcLength(cnt,True)
                approx=cv2.approxPolyDP(cnt,0.018*peri,True)
                if 4<=len(approx)<=6:
                    x,y,bw,bh=cv2.boundingRect(approx)
                    if bh==0: continue
                    ar=bw/float(bh)
                    if 1.5<=ar<=7 and bw>50 and bh>12:
                        s=4+(3 if y>h*0.4 else 0)+(2 if len(approx)==4 else 0)
                        c.append({'x1':x,'y1':y,'x2':x+bw,'y2':y+bh,'score':s,'contour':approx if len(approx)==4 else None,'method':'contour'})
    return c

def find_ocr_plates(image, reader):
    h,w=image.shape[:2]
    sc=min(500/w,1.0)
    small=cv2.resize(image,None,fx=sc,fy=sc) if sc<1 else image.copy()
    results=reader.readtext(small,decoder='greedy',paragraph=False,min_size=10,text_threshold=0.3,low_text=0.3,width_ths=0.8)
    cands=[]; scan_texts=[]
    for(bbox,text,conf) in results:
        cl=re.sub(r'^IND','',re.sub(r'[^A-Za-z0-9]','',text.upper()))
        if len(cl)<2:
            continue
        scan_texts.append((cl,conf))
        pts=np.array(bbox)
        x1=int(pts[:,0].min()/sc); y1=int(pts[:,1].min()/sc); x2=int(pts[:,0].max()/sc); y2=int(pts[:,1].max()/sc)
        s=2
        if re.search(r'[A-Z]',cl) and re.search(r'\d',cl): s+=3
        if len(cl)>=6: s+=2
        if len(cl)>=8: s+=2
        if re.match(r'[A-Z]{2}\d',cl): s+=3
        cands.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2,'score':s,'contour':None,'method':'ocr','text':cl})
    cands.sort(key=lambda b:b['x1'])
    merged=[]
    for box in cands:
        if merged:
            last=merged[-1]
            gap=box['x1']-last['x2']
            ydiff=abs((last['y1']+last['y2'])//2-(box['y1']+box['y2'])//2)
            havg=((last['y2']-last['y1'])+(box['y2']-box['y1']))/2
            if havg>0 and gap<havg*2.5 and ydiff<havg*0.7:
                merged[-1]={
                    'x1':min(last['x1'],box['x1']),'y1':min(last['y1'],box['y1']),
                    'x2':max(last['x2'],box['x2']),'y2':max(last['y2'],box['y2']),
                    'score':last['score']+box['score']+2,'contour':None,'method':'ocr_merged',
                    'text':last.get('text','')+box.get('text','')
                }
                continue
        merged.append(box.copy())
    return merged,scan_texts

def find_plate(image, plate_model, reader):
    h,w=image.shape[:2]
    t=time.time()
    yolo_c=find_plate_yolo(image,plate_model)
    ty=time.time()-t
    yolo_conf=yolo_c[0]['yolo_conf'] if yolo_c else 0
    scan_texts=[]

    if yolo_c and yolo_c[0]['score']>8:
        print(f"   ⚡ YOLO ({ty:.2f}s, conf:{yolo_conf:.0%})")
        all_c=yolo_c
    else:
        print(f"   ⚠️ YOLO weak ({ty:.2f}s) — fallbacks...")
        white_c=find_white_plates(image)
        contour_c=find_contour_plates(image)
        ocr_c,scan_texts=find_ocr_plates(image,reader)
        all_c=yolo_c+white_c+contour_c+ocr_c

    all_c.append({'x1':0,'y1':int(h*0.55),'x2':w,'y2':h,'score':0,'contour':None,'method':'bottom'})
    all_c.sort(key=lambda c:c['score'],reverse=True)

    unique=[]
    for c in all_c:
        dup=False
        for u in unique:
            ix1=max(c['x1'],u['x1']); iy1=max(c['y1'],u['y1']); ix2=min(c['x2'],u['x2']); iy2=min(c['y2'],u['y2'])
            if ix2>ix1 and iy2>iy1:
                if (ix2-ix1)*(iy2-iy1)/max((c['x2']-c['x1'])*(c['y2']-c['y1']),1)>0.4:
                    dup=True; break
        if not dup:
            unique.append(c)

    crops=[]
    for cand in unique[:1]:
        px=max(25,int((cand['x2']-cand['x1'])*0.12))
        py=max(15,int((cand['y2']-cand['y1'])*0.25))
        x1=max(0,cand['x1']-px); y1=max(0,cand['y1']-py); x2=min(w,cand['x2']+px); y2=min(h,cand['y2']+py)
        crop=image[y1:y2,x1:x2].copy()
        if crop.shape[0]>10 and crop.shape[1]>20:
            crops.append({'crop':crop,'bbox':(x1,y1,x2,y2),'contour':cand.get('contour',None),'score':cand['score'],'method':cand['method']})
    return crops,scan_texts,yolo_conf

# ═══════════════════════════════════════════
# PREPROCESSING + OCR
# ═══════════════════════════════════════════

def order_points(pts):
    pts=pts.reshape(4,2).astype(np.float32)
    rect=np.zeros((4,2),dtype=np.float32)
    s=pts.sum(axis=1)
    rect[0]=pts[np.argmin(s)]
    rect[2]=pts[np.argmax(s)]
    d=np.diff(pts,axis=1)
    rect[1]=pts[np.argmin(d)]
    rect[3]=pts[np.argmax(d)]
    return rect

def perspective_correct(image,contour):
    try:
        rect=order_points(contour)
        tl,tr,br,bl=rect
        maxW=max(int(np.linalg.norm(br-bl)),int(np.linalg.norm(tr-tl)))
        maxH=max(int(np.linalg.norm(tr-br)),int(np.linalg.norm(tl-bl)))
        if maxW<20 or maxH<10:
            return image
        dst=np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]],dtype=np.float32)
        return cv2.warpPerspective(image,cv2.getPerspectiveTransform(rect,dst),(maxW,maxH))
    except:
        return image

def deskew(image):
    if len(image.shape)==3:
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    else:
        gray=image.copy()
    edges=cv2.Canny(gray,50,150)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength=30,maxLineGap=10)
    if lines is not None:
        angles=[np.degrees(np.arctan2(l[0][3]-l[0][1],l[0][2]-l[0][0])) for l in lines if abs(np.degrees(np.arctan2(l[0][3]-l[0][1],l[0][2]-l[0][0])))<15]
        if angles:
            angle=np.median(angles)
            if abs(angle)>0.5:
                h,w=image.shape[:2]
                M=cv2.getRotationMatrix2D((w//2,h//2),angle,1.0)
                image=cv2.warpAffine(image,M,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    return image

def preprocess_plate(plate_img,contour_4=None):
    h,w=plate_img.shape[:2]
    if h<5 or w<10:
        return {'dummy':np.ones((100,300),dtype=np.uint8)*255},np.ones((100,300,3),dtype=np.uint8)*255

    if contour_4 is not None:
        try:
            plate_img=perspective_correct(plate_img,contour_4)
            h,w=plate_img.shape[:2]
        except:
            pass

    plate_img=deskew(plate_img)
    th=500
    s=th/h
    nw=int(w*s)
    big=cv2.resize(plate_img,(nw,th),interpolation=cv2.INTER_LANCZOS4)

    if len(big.shape)==3:
        gray=cv2.cvtColor(big,cv2.COLOR_BGR2GRAY)
    else:
        gray=big.copy()
        big=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)

    by=int(th*0.07); bx=int(nw*0.02)

    def cb(i):
        i=i.copy()
        if len(i.shape)==2:
            i[:by,:]=255; i[-by:,:]=255; i[:,:bx]=255; i[:,-bx:]=255
        return i

    gray=cb(gray)
    clahe=cv2.createCLAHE(clipLimit=4.0,tileGridSize=(8,8))
    v={}

    rh=300
    rs=rh/plate_img.shape[0]
    rw=int(plate_img.shape[1]*rs)
    v['raw_crop']=cv2.resize(plate_img,(rw,rh),interpolation=cv2.INTER_CUBIC)

    v1=clahe.apply(gray)
    for _ in range(1):
        g=cv2.GaussianBlur(v1,(0,0),2)
        v1=cv2.addWeighted(v1,2.3,g,-1.3,0)
    v['sharp_gray']=np.clip(v1,0,255).astype(np.uint8)

    smooth=cv2.bilateralFilter(gray,9,75,75)
    _,v3=cv2.threshold(smooth,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    v['otsu_bilateral']=cb(v3)

    return v,big

def ocr_easyocr(img,reader,label):
    try:
        if len(img.shape)==2:
            img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        return [
            (re.sub(r'^IND','',re.sub(r'[^A-Za-z0-9]','',t.upper())), c, f'easy_{label}')
            for(_,t,c) in reader.readtext(
                img,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                paragraph=False,
                min_size=5,
                text_threshold=0.35,
                low_text=0.2,
                width_ths=0.5,
                decoder='greedy'
            ) if len(re.sub(r'[^A-Za-z0-9]','',t))>=2
        ]
    except:
        return []

def ocr_tesseract(img,psm,label):
    try:
        if len(img.shape)==3:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cfg=f'--oem 3 --psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        data=pytesseract.image_to_data(img,config=cfg,output_type=pytesseract.Output.DICT)
        texts=[]
        for i in range(len(data['text'])):
            w=re.sub(r'^IND','',re.sub(r'[^A-Za-z0-9]','',data['text'][i].upper()))
            c=int(data['conf'][i]) if str(data['conf'][i])!='-1' else 0
            if len(w)>=2 and c>10:
                texts.append((w,c/100.0,f'tess_{label}_p{psm}'))
        return texts
    except:
        return []

def fix_state(s):
    s=s.upper()
    if s in VALID_STATES:
        return s,True
    if s in STATE_FIXES and STATE_FIXES[s] in VALID_STATES:
        return STATE_FIXES[s],True
    for p in [0,1]:
        ch=s[p]
        if ch in SIMILAR:
            for a in SIMILAR[ch]:
                if len(a)!=1:
                    continue
                t=(a+s[1]) if p==0 else (s[0]+a)
                if t in VALID_STATES:
                    return t,True
        if ch in TO_LETTER:
            t=(TO_LETTER[ch]+s[1]) if p==0 else (s[0]+TO_LETTER[ch])
            if t in VALID_STATES:
                return t,True
    return s,False

def fix_plate(raw):
    t=re.sub(r'^IND','',re.sub(r'[^A-Za-z0-9]','',raw.upper()))
    if len(t)<7:
        return t,0.0,99

    formats=['LLDDLLDDDD','LLDDLDDDD']
    best=t; best_score=0.0; best_corr=99

    for fmt in formats:
        fl=len(fmt)
        if len(t)<fl:
            continue

        attempt=t[:fl]
        ch=list(attempt)
        corr=0
        sc=0.0

        for i,e in enumerate(fmt):
            c=ch[i]
            if e=='L':
                if c.isalpha():
                    sc+=1.0
                elif c in TO_LETTER:
                    ch[i]=TO_LETTER[c]
                    sc+=0.35
                    corr+=1
            else:
                if c.isdigit():
                    sc+=1.0
                elif c.upper() in TO_DIGIT:
                    ch[i]=TO_DIGIT[c.upper()]
                    sc+=0.35
                    corr+=1

        st=''.join(ch[:2])
        fs,ok=fix_state(st)
        if st!=fs:
            corr+=1
        ch[0],ch[1]=fs[0],fs[1]
        if ok:
            sc+=2.0

        out=''.join(ch)
        n=sc/(fl+2.0)

        if validate_plate(out):
            n += 0.20

        if n>best_score or (n==best_score and corr<best_corr):
            best=out; best_score=n; best_corr=corr

    return best,best_score,best_corr

def validate_plate(t):
    if len(t)==10:
        return t[0:2].isalpha() and t[2:4].isdigit() and t[4:6].isalpha() and t[6:10].isdigit()
    elif len(t)==9:
        return t[0:2].isalpha() and t[2:4].isdigit() and t[4].isalpha() and t[5:9].isdigit()
    return False

def format_plate(t):
    if len(t)==10 and validate_plate(t):
        return f"{t[0:2]} {t[2:4]} {t[4:6]} {t[6:10]}"
    elif len(t)==9 and validate_plate(t):
        return f"{t[0:2]} {t[2:4]} {t[4]} {t[5:9]}"
    return t

def merge_fragments(texts):
    merged=list(texts); groups={}
    for t,c,s in texts:
        k=s.split('_')[1] if '_' in s else 'x'
        groups.setdefault(k,[]).append((t,c,s))
    for k,items in groups.items():
        if len(items)<2:
            continue
        for i in range(len(items)):
            for j in range(i+1,len(items)):
                t1,c1,_=items[i]; t2,c2,_=items[j]
                if len(t1)>=8 or len(t2)>=8:
                    continue
                for combo in [t1+t2,t2+t1]:
                    if 8<=len(combo)<=12:
                        merged.append((combo,(c1+c2)/2*0.75,f'merged_{k}'))
    return merged

def read_plate_all(variants, reader, is_primary=True):
    at=[]
    # primary raw crop is most important
    if 'raw_crop' in variants:
        at.extend([(t,c,'easy_raw_crop_primary') for t,c,_ in ocr_easyocr(variants['raw_crop'], reader, 'raw_crop')])
    if 'sharp_gray' in variants:
        at.extend(ocr_tesseract(variants['sharp_gray'], 7, 'sharp_gray'))
    if 'otsu_bilateral' in variants:
        at.extend(ocr_tesseract(variants['otsu_bilateral'], 7, 'otsu_bilateral'))
    return at

def pick_best_ocr(all_texts, scan_texts):
    combined = list(all_texts)
    for t,c in scan_texts:
        combined.append((t, c*0.90, 'scan'))

    combined = merge_fragments(combined)
    if not combined:
        return "", 0.0, combined

    candidates = []

    for raw, conf, source in combined:
        raw_clean = clean_plate_text(raw)
        if len(raw_clean) < 4:
            continue

        fixed, fmt_score, corr = fix_plate(raw_clean)
        raw_valid = validate_plate(raw_clean)
        fixed_valid = validate_plate(fixed)

        # prefer raw if already valid
        chosen = raw_clean if raw_valid else fixed
        chosen_valid = validate_plate(chosen)

        # stronger penalty for too many corrections
        corr_pen = 0.18 if corr == 0 else (0.08 if corr == 1 else (-0.02 if corr == 2 else -0.15))

        src_bonus = 0.08 if 'primary' in source else (0.04 if 'easy' in source else 0.02)

        score = (
            conf * 0.35 +
            fmt_score * 0.20 +
            (0.25 if chosen_valid else 0.0) +
            (0.10 if len(chosen) in [9,10] else 0.0) +
            (0.08 if chosen[:2] in VALID_STATES else 0.0) +
            corr_pen +
            src_bonus
        )

        # if raw is valid and fixed differs, prefer raw strongly
        if raw_valid and raw_clean != fixed:
            score += 0.12

        candidates.append((chosen, score, conf, source, raw_clean, corr, raw_valid, fixed_valid))

    if not candidates:
        best = max(combined, key=lambda x:x[1])
        return best[0], best[1], combined

    valid_candidates = [c for c in candidates if validate_plate(c[0])]
    if valid_candidates:
        candidates = valid_candidates

    # vote support
    vote_count = {}
    for c in candidates:
        plate = c[0]
        vote_count[plate] = vote_count.get(plate, 0) + 1

    rescored = []
    for c in candidates:
        plate, score, conf, source, raw_clean, corr, raw_valid, fixed_valid = c
        score += min(vote_count.get(plate,0)*0.03, 0.12)
        rescored.append((plate, score, conf, source, raw_clean, corr))

    rescored.sort(key=lambda x:x[1], reverse=True)
    best = rescored[0]
    return best[0], min(best[1], 1.0), combined

# ═══════════════════════════════════════════
# MATCH ENGINE
# ═══════════════════════════════════════════

def evaluate_single_match(ocr_plate, db_plate):
    ocr = clean_plate_text(ocr_plate)
    db = clean_plate_text(db_plate)

    full_matches, full_diffs, full_diff_pos, full_total = exact_position_compare(ocr, db)
    exact_full = (ocr == db)

    last6_ocr, last6_db, last6_matches, last6_diffs, last6_diff_pos, last6_total = suffix_position_compare(ocr, db, 6)
    last4_ocr, last4_db, last4_matches, last4_diffs, last4_diff_pos, last4_total = suffix_position_compare(ocr, db, 4)
    last5_ocr, last5_db, last5_matches, last5_diffs, last5_diff_pos, last5_total = suffix_position_compare(ocr, db, 5)

    lcs_text_6, lcs_len_6 = longest_common_substring(last6_ocr, last6_db)
    lcs_text_4, lcs_len_4 = longest_common_substring(last4_ocr, last4_db)
    lcs_text_5, lcs_len_5 = longest_common_substring(last5_ocr, last5_db)

    mid_alpha_match = False
    if len(ocr) >= 6 and len(db) >= 6:
        mid_alpha_match = (
            ocr[4:6].isalpha() and db[4:6].isalpha() and
            (ocr[4:6] == db[4:6])
        )

    stage = "NO MATCH"
    stage_conf = 0.0
    stage5_reason = ""

    if exact_full:
        stage = "STAGE 1 - FULL EXACT MATCH"
        stage_conf = 1.00
    elif len(ocr) == len(db) and full_diffs in [1,2]:
        stage = "STAGE 2 - FULL POSITION MATCH (1-2 MISMATCH)"
        stage_conf = max(0.90, full_matches / max(full_total,1))
    elif last6_total > 0 and (last6_matches >= 5 or last6_diffs <= 1 or lcs_len_6 >= 5):
        stage = "STAGE 3 - LAST 6 MATCH"
        stage_conf = max((last6_matches / max(last6_total,1))*0.85 + (lcs_len_6 / max(last6_total,1))*0.15, 0.80)
    elif last4_total > 0 and (last4_matches >= 3 or last4_diffs <= 1 or lcs_len_4 >= 3):
        stage = "STAGE 4 - LAST 4 MATCH"
        stage_conf = max((last4_matches / max(last4_total,1))*0.85 + (lcs_len_4 / max(last4_total,1))*0.15, 0.70)
    elif last5_total > 0:
        if lcs_len_5 >= 3:
            stage = "STAGE 5 - SERIES SIMILARITY (LAST 5)"
            stage_conf = max((lcs_len_5 / max(last5_total,1)), 0.60)
            stage5_reason = "3+ series in last5"
        elif lcs_len_5 == 2 and mid_alpha_match:
            stage = "STAGE 5 - SERIES SIMILARITY (LAST 5 + 5-6 ALPHA)"
            base = (lcs_len_5 / max(last5_total,1))
            stage_conf = min(max(base + 0.20, 0.55), 0.75)
            stage5_reason = "2 series + 5th/6th alphabets match"

    overall_conf = stage_conf

    return {
        'db_plate': db_plate,
        'ocr_plate': ocr_plate,
        'exact_full': exact_full,
        'full_matches': full_matches,
        'full_diffs': full_diffs,
        'full_diff_positions': full_diff_pos,
        'full_total': full_total,
        'last6_ocr': last6_ocr,
        'last6_db': last6_db,
        'last6_matches': last6_matches,
        'last6_diffs': last6_diffs,
        'last6_total': last6_total,
        'last4_ocr': last4_ocr,
        'last4_db': last4_db,
        'last4_matches': last4_matches,
        'last4_diffs': last4_diffs,
        'last4_total': last4_total,
        'last5_ocr': last5_ocr,
        'last5_db': last5_db,
        'last5_matches': last5_matches,
        'last5_diffs': last5_diffs,
        'last5_total': last5_total,
        'lcs_text_6': lcs_text_6,
        'lcs_len_6': lcs_len_6,
        'lcs_text_4': lcs_text_4,
        'lcs_len_4': lcs_len_4,
        'lcs_text_5': lcs_text_5,
        'lcs_len_5': lcs_len_5,
        'mid_alpha_match': mid_alpha_match,
        'stage5_reason': stage5_reason,
        'stage': stage,
        'stage_conf': stage_conf,
        'overall_conf': overall_conf
    }

def stage_priority(stage_name):
    order = {
        "STAGE 1 - FULL EXACT MATCH": 1,
        "STAGE 2 - FULL POSITION MATCH (1-2 MISMATCH)": 2,
        "STAGE 3 - LAST 6 MATCH": 3,
        "STAGE 4 - LAST 4 MATCH": 4,
        "STAGE 5 - SERIES SIMILARITY (LAST 5)": 5,
        "STAGE 5 - SERIES SIMILARITY (LAST 5 + 5-6 ALPHA)": 5,
        "NO MATCH": 99
    }
    return order.get(stage_name, 99)

def get_all_sheet_matches(ocr_plate, ocr_texts):
    all_candidates = [(ocr_plate, 1.0)]
    all_raw = []

    for raw, conf, src in ocr_texts:
        rc = re.sub(r'^IND', '', re.sub(r'[^A-Za-z0-9]', '', str(raw).upper()))
        all_raw.append(rc)
        if len(rc) >= 4:
            all_candidates.append((rc, conf))
        fixed, _, _ = fix_plate(rc)
        if fixed != rc and len(fixed) >= 4:
            all_candidates.append((fixed, conf))

    matches = []
    seen = set()

    for ct, cc in all_candidates:
        for db_plate in PLATE_DB_LIST:
            key = (ct, db_plate)
            if key in seen:
                continue
            seen.add(key)

            res = evaluate_single_match(ct, db_plate)
            if res['stage'] != "NO MATCH":
                res['ocr_source_conf'] = cc
                matches.append(res)

    matches.sort(
        key=lambda x: (
            stage_priority(x['stage']),
            -x['overall_conf'],
            -x['full_matches'],
            -x['last6_matches'],
            -x['last4_matches'],
            -x['lcs_len_5']
        )
    )

    final_matches = []
    used = set()
    for m in matches:
        if m['db_plate'] not in used:
            final_matches.append(m)
            used.add(m['db_plate'])

    return final_matches[:5]

def print_table(ocr_plate, ocr_conf, yolo_conf, sheet_matches, decision, justification):
    html = """<style>
    .pt{border-collapse:collapse;width:100%;font-family:monospace;margin:10px 0}
    .pt th{background:#2d2d2d;color:white;padding:8px;text-align:left;border:1px solid #555;font-size:12px}
    .pt td{padding:6px 8px;border:1px solid #ddd;font-size:12px}
    .sel{background:#c8e6c9!important;font-weight:bold}
    .ocr{background:#e3f2fd!important}
    .db{background:#f0f0f0;border:2px solid #333;border-radius:8px;padding:15px;margin:10px 0}
    .dt{font-size:15px;font-weight:bold;margin-bottom:8px}
    </style>"""

    html += '<table class="pt"><tr>'
    html += '<th>#</th><th>Source</th><th>Plate</th><th>Conf</th><th>Confirmed At</th><th>Full</th><th>Last6</th><th>Last4</th><th>Series Last5</th><th>Decision</th></tr>'

    is_ocr = (decision == "OCR")
    rc = 'sel' if is_ocr else 'ocr'
    mark = '◀ SELECTED' if is_ocr else ''
    html += f'<tr class="{rc}"><td>1</td><td>🟢 YOLO+OCR</td><td><b>{format_plate(ocr_plate)}</b></td>'
    html += f'<td>YOLO:{yolo_conf:.0%} | OCR:{ocr_conf:.0%}</td><td>OCR</td><td>—</td><td>—</td><td>—</td><td>—</td><td><b>{mark}</b></td></tr>'

    for i, m in enumerate(sheet_matches):
        db = m['db_plate']
        is_sel = (decision == "SHEET" and i == 0)
        rc = 'sel' if is_sel else ''
        mark = '◀ SELECTED' if is_sel else ''
        html += f'<tr class="{rc}"><td>{i+2}</td><td>📊 Sheet</td><td><b>{format_plate(db)}</b></td>'
        html += f'<td>{m["overall_conf"]:.0%}</td>'
        html += f'<td>{m["stage"]}</td>'
        html += f'<td>{m["full_matches"]}/{m["full_total"]} (d:{m["full_diffs"]})</td>'
        html += f'<td>{m["last6_matches"]}/{m["last6_total"]} (d:{m["last6_diffs"]})</td>'
        html += f'<td>{m["last4_matches"]}/{m["last4_total"]} (d:{m["last4_diffs"]})</td>'
        html += f'<td>{m["lcs_text_5"]} ({m["lcs_len_5"]})</td>'
        html += f'<td><b>{mark}</b></td></tr>'

    html += '</table>'
    sel = format_plate(ocr_plate if decision == "OCR" else (sheet_matches[0]['db_plate'] if sheet_matches else ocr_plate))
    html += f'<div class="db"><div class="dt">🏆 SELECTED: {sel}</div><b>Justification:</b> {justification}</div>'
    display(HTML(html))

def detect_plate(image_path=None, image_array=None):
    global LAST_PLATE_BBOX

    start = time.time()
    if image_array is not None:
        image = image_array.copy()
    else:
        image = cv2.imread(image_path)

    if image is None:
        print("❌ Cannot load!")
        return None

    mw = 1024
    if image.shape[1] > mw:
        s = mw / image.shape[1]
        image = cv2.resize(image, None, fx=s, fy=s)

    original = image.copy()

    t1 = time.time()
    print("🔍 Phase 1: Finding plate...")
    crops, scan_texts, yolo_conf = find_plate(image, plate_model, reader)
    print(f"   ⏱️ {time.time()-t1:.1f}s")

    t2 = time.time()
    print(f"\n🔧📖 Phase 2+3: Preprocess + OCR ({len(crops)} candidates)...")
    all_plate_texts = []
    best_variants = None
    for idx, cand in enumerate(crops):
        is_primary = (idx == 0)
        variants, big = preprocess_plate(cand['crop'], cand.get('contour'))
        if is_primary:
            best_variants = variants
        texts = read_plate_all(variants, reader, is_primary=is_primary)
        all_plate_texts.extend(texts)
        print(f"   📷 Cand {idx+1} [{cand['method']}]: {len(texts)} reads")
    print(f"   ⏱️ {time.time()-t2:.1f}s")

    t3 = time.time()
    ocr_plate, ocr_conf, all_combined = pick_best_ocr(all_plate_texts, scan_texts)
    print(f"\n🗳️ Phase 4: OCR result: {format_plate(ocr_plate)} (YOLO:{yolo_conf:.0%} OCR:{ocr_conf:.0%}) ⏱️{time.time()-t3:.3f}s")

    final_plate = ocr_plate
    final_conf = ocr_conf
    decision = "OCR"
    justification = "No spreadsheet stage passed. Showing OCR + YOLO result."
    sheet_matches = []
    vehicle_info = None

    if PLATE_DB and len(PLATE_DB) > 0:
        t5 = time.time()
        print(f"\n📊 Phase 5: Spreadsheet matching ({len(PLATE_DB)} plates)...")
        all_ocr_for_match = all_plate_texts + [(t, c, 'scan') for t, c in scan_texts]
        sheet_matches = get_all_sheet_matches(ocr_plate, all_ocr_for_match)

        if sheet_matches:
            top = sheet_matches[0]
            print(f"\n   ✅ Best spreadsheet candidate: {format_plate(top['db_plate'])}")
            print(f"   ✅ Confirmed at: {top['stage']}")
            print(f"   ✅ Stage confidence: {top['overall_conf']:.0%}")

            if top['stage'] != "NO MATCH":
                decision = "SHEET"
                final_plate = top['db_plate']
                final_conf = min(top['overall_conf'] + 0.05, 1.0)
                found, info = search_plate_in_sheet(top['db_plate'])
                if found:
                    vehicle_info = info
                justification = (
                    f"Selected from spreadsheet. Confirmed at {top['stage']}. "
                    f"Stage confidence={top['overall_conf']:.0%}."
                )
        else:
            decision = "OCR"
            final_plate = ocr_plate
            final_conf = ocr_conf
            justification = (
                f"No spreadsheet stage passed till Stage 5. "
                f"Showing OCR + YOLO only. YOLO={yolo_conf:.0%}, OCR={ocr_conf:.0%}."
            )

        print(f"   ⏱️ {time.time()-t5:.3f}s")
    else:
        justification = "No spreadsheet loaded. Showing OCR + YOLO result."

    formatted = format_plate(final_plate)
    is_valid = validate_plate(final_plate)
    total = time.time() - start

    annotated = original.copy()
    if crops:
        x1,y1,x2,y2 = crops[0]['bbox']
        LAST_PLATE_BBOX = crops[0]['bbox']
        color = (0,200,0) if decision=="SHEET" or (decision=="OCR" and is_valid) else (0,140,255)
        cv2.rectangle(annotated,(x1,y1),(x2,y2),color,3)
        font=cv2.FONT_HERSHEY_SIMPLEX
        ts=cv2.getTextSize(formatted,font,1.4,3)[0]
        cv2.rectangle(annotated,(x1,y1-ts[1]-30),(x1+ts[0]+20,y1-5),color,-1)
        cv2.putText(annotated,formatted,(x1+10,y1-15),font,1.4,(255,255,255),3)
    else:
        LAST_PLATE_BBOX = None

    fig=plt.figure(figsize=(24,10))
    ax1=fig.add_subplot(2,5,(1,2))
    ax1.imshow(cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB))
    ax1.set_title('🚗 RESULT',fontsize=15,fontweight='bold')
    ax1.axis('off')

    if crops:
        ax2=fig.add_subplot(2,5,3)
        c0=crops[0]['crop']
        if len(c0.shape)==3:
            ax2.imshow(cv2.cvtColor(c0,cv2.COLOR_BGR2RGB))
        else:
            ax2.imshow(c0,cmap='gray')
        ax2.set_title(f"Crop [{crops[0]['method']}]",fontsize=11)
        ax2.axis('off')

    if best_variants:
        dv=[(n,v) for n,v in best_variants.items() if n!='raw_crop']
        for i in range(min(2,len(dv))):
            ax=fig.add_subplot(2,5,4+i)
            img=dv[i][1]
            if len(img.shape)==3:
                ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(img,cmap='gray')
            ax.set_title(dv[i][0],fontsize=10)
            ax.axis('off')

    plt.suptitle(f'🏆 {formatted}  |  {final_conf:.0%}  |  {total:.1f}s',fontsize=16,fontweight='bold',y=1.02)
    plt.tight_layout()
    plt.show()

    print(f"\n{'═'*70}")
    print("  📊 COMPARISON TABLE")
    print(f"{'═'*70}")
    print_table(ocr_plate, ocr_conf, yolo_conf, sheet_matches, decision, justification)

    print()
    print("┏"+"━"*70+"┓")
    print(f"┃  🚗  PLATE:          {formatted:<46} ┃")
    print(f"┃  🟣  YOLO conf:      {yolo_conf:.0%}{' '*49} ┃")
    print(f"┃  🟢  OCR conf:       {ocr_conf:.0%}{' '*49} ┃")
    print(f"┃  📊  FINAL conf:     {final_conf:.0%}{' '*49} ┃")
    print(f"┃  ⏱️   TIME:           {total:.1f}s{' '*49} ┃")
    print(f"┃  📐  FORMAT:         {'✅ Valid' if is_valid else '❌ Invalid'}{' '*44} ┃")
    src = "📊 Spreadsheet" if decision=="SHEET" else "🟢 YOLO+OCR"
    print(f"┃  🔀  SOURCE:         {src:<46} ┃")
    print("┗"+"━"*70+"┛")

    print(f"\n📋 OCR reads:")
    seen=set()
    for raw,conf,source in sorted(all_combined,key=lambda x:-x[1]):
        fixed,_,corr=fix_plate(raw)
        if fixed not in seen and len(fixed)>=4:
            e="🟢" if 'easy' in source else("🔵" if 'tess' in source else "⚪")
            win=" ◀ OCR" if fixed==ocr_plate else ""
            print(f"   {e} '{raw:15s}' → '{format_plate(fixed):15s}' {conf:.0%}{win}")
            seen.add(fixed)

    global LAST_PLATE_BBBOX
    LAST_PLATE_BBBOX = crops[0]['bbox'] if crops else None

    return formatted, final_conf, annotated, final_plate

print("✅ Updated anti-mismatch Cell 3 ready!")
!pip install moondream einops -q

import cv2
import numpy as np
import re
import time
import matplotlib.pyplot as plt
import easyocr

# Initialize EasyOCR (Local & Offline)
reader = easyocr.Reader(['en'], gpu=False)

RESULTS_LOG = []
LAST_PLATE_BBOX = None

def clean_slot_text(t):
    """Cleans text: No hyphen, fixes G12, and handles common OCR swaps."""
    if not t: return ""

    # 1. Basic cleaning: Upper case and strip
    t = str(t).upper().strip()

    # 2. Fix common character swaps (OCR often sees G as 6 or 0)
    # We do these replacements to help the regex find the pattern
    t = t.replace('S', '5').replace('O', '0')

    # 3. Regex: Look for a Letter/Number followed by numbers (allows space/hyphen in between)
    # Pattern: [Letter or Digit] [Optional Space/Hyphen] [1 or 2 Digits]
    match = re.search(r'([A-Z0-9])\s*[-]?\s*(\d{1,2})', t)

    if match:
        char_part = match.group(1)
        num_part = match.group(2)

        # Correction: If the first character is '6' or '0', it is almost certainly 'G'
        if char_part in ['6', '0']:
            char_part = 'G'

        # Return combined WITHOUT hyphen (e.g., G12, M14)
        return f"{char_part}{num_part}"

    return ""

def get_slot_roi(image, plate_bbox=None):
    """Crops the floor area based on plate position."""
    h, w = image.shape[:2]
    if plate_bbox is not None:
        x1, y1, x2, y2 = plate_bbox
        pw, ph = (x2 - x1), (y2 - y1)
        # Wide crop to ensure we don't cut off the 'G'
        rx1 = max(0, x1 - int(pw * 1.5))
        rx2 = min(w, x2 + int(pw * 1.5))
        ry1 = min(h, y2 + int(ph * 1.0))
        ry2 = min(h, y2 + int(ph * 9.0))
        return image[ry1:ry2, rx1:rx2].copy()

    # Default fallback
    return image[int(h*0.7):h, int(w*0.1):int(w*0.9)].copy()

def detect_floor_slot(image_path=None, image_array=None, show=True, plate_bbox=None):
    """OCR Engine: Uses LAB Yellow Boost for best accuracy."""
    t0 = time.time()
    img = image_array.copy() if image_array is not None else cv2.imread(image_path)
    if img is None: return "", 0.0, None

    # 1. Get Floor ROI
    roi = get_slot_roi(img, plate_bbox=plate_bbox)

    # 2. IMAGE ENHANCEMENT (LAB Yellow Boost)
    # Digital Zoom
    enhanced = cv2.resize(roi, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    # LAB Boost: Separates Yellow paint from Gray concrete
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    b = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
    enhanced_final = cv2.merge([l, a, b])
    enhanced_final = cv2.cvtColor(enhanced_final, cv2.COLOR_LAB2BGR)
    # Slight blur to reduce "grain" noise from the floor
    enhanced_final = cv2.GaussianBlur(enhanced_final, (3,3), 0)

    # 3. Perform OCR
    # contrast_ths=0.1 and low_text=0.3 help read faded/faint paint
    results = reader.readtext(enhanced_final, detail=0, contrast_ths=0.1, low_text=0.3)

    best_slot = "N/A"
    for res in results:
        cleaned = clean_slot_text(res)
        if len(cleaned) >= 2: # Found a valid G12, M14, etc.
            best_slot = cleaned
            break

    if show:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1); plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)); plt.title("Floor Area")
        plt.subplot(1, 2, 2); plt.imshow(cv2.cvtColor(enhanced_final, cv2.COLOR_BGR2RGB)); plt.title(f"OCR: {best_slot}")
        plt.show()
        print(f" 🅿️  Slot Detected: {best_slot} | Time: {time.time()-t0:.1f}s")

    return best_slot, 1.0, roi

print("✅ Cell 4 Restored: LAB Enhancement + G12 Fix + No Hyphen is ready.")
