import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime

COMMON_DATE_FORMATS=[
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%Y/%m/%d",
    "%Y-%m-%d",
    "%Y.%m.%d",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%m.%d.%Y"
]

def load_data(file):
    #Universal data loader for CSV datasets.
    #Handles different seperators, encoding issues and corrupted rows.
    try:
        data=pd.read_csv(
            file,
            sep=None,
            engine="python",
            encoding="utf-8",
            on_bad_lines="warn",
            #nrows=10
            )
    except UnicodeDecodeError:
        data=pd.read_csv(
            file,
            sep=None,
            engine="python",
            encoding="latin1",
            on_bad_lines="warn",
            #nrows=10
            )
    except Exception as e:
        print(f"Failed to load data: {e}.")
        return None
    print(f"Successfully loaded dataset with shape: {data.shape}.")
    return data

def validate_data(data):
    if data is None:
        raise ValueError("Dataset failed to load.")
    if data.empty:
        raise ValueError("Dataset is empty.")
    print("Data validation passed.")

def clean_data(data):
    #Universal data cleaning pipeline
    data=data.copy()
    print("Starting data cleaning pipeline.")
    data=standardise_column_names(data)
    data=detect_and_parse_datetime(data)
    data,metadata=intelligent_preprocess(data)
    #print(metadata)
    data=remove_outliers(data)
    print("Data cleaning complete.")
    return data,metadata

def standardise_column_names(data):
    #makes system dataset-independent
    data.columns=(
        data.columns
        .str.strip()
        .str.lower()
        .str.replace(" ","_")
        .str.replace(r"[^\w_]","",regex=True)
    )
    return data

def detect_and_parse_datetime(df):
    #Universal datetime detection and parsing pipeline.
    #Handles combined datetime columns, separate date+time columsn, date-only columns and time-only columns
    df=df.copy()
    cols=df.columns
    date_col=None
    time_col=None
    #Detect date and time columns
    for col in cols:
        if "date" in col:
            date_col=col
        elif "time" in col:
            time_col=col
    #Combine date and time
    if date_col and time_col:
        try:
            dt_series=df[date_col].astype(str)+" "+df[time_col].astype(str)
            df["datetime"]=pd.to_datetime(
                dt_series,
                dayfirst=True,
            )
            df.drop(columns=[date_col,time_col],inplace=True)
            print(f"Combined '{date_col}' + '{time_col}'")
            return df
        except Exception as e:
            print(f"Failed to combined date + time: {e}.")
    #Detect full datetime or date-only column
    if date_col:
        for col in cols:
            if pd.api.types.is_object_dtype(df[col]):
                for fmt in COMMON_DATE_FORMATS:
                    try:
                        parsed=pd.to_datetime(df[col],format=fmt)
                        df.drop(columns=[col],inplace=True)
                        df["datetime"]=parsed
                        print(f"Parsed date column '{col}' using format {fmt}.")
                        return df
                    except:
                        continue
                #Infer datetime format
                try:
                    parsed=pd.to_datetime(
                        df[col],
                        dayfirst=True,
                    )
                    df.drop(columns=[col],inplace=True)
                    df["datetime"]=parsed
                    print(f"Parsed datetime column: {col}.")
                    return df
                except:
                    continue
    #Time-only handling
    if time_col:
        try:
            t=pd.to_datetime(df[time_col],format="%H:%M:%S",errors="coerce")
            if t.isnull().all():
                #Fallback to generic parsing
                t=pd.to_datetime(df[time_col],errors="coerce")
            if t.isnull().all():
                print("Cannot parse time-only column.")
                return df
            #Detect day rollover
            day_counter=0
            continuous_time=[]
            prev_seconds=t.dt.hour*3600+t.dt.minute*60+t.dt.second
            for s in prev_seconds:
                #new day if time decreased
                if continuous_time and s<continuous_time[-1]%86400:
                    day_counter+=1
                #total seconds=seconds+86400*days
                continuous_time.append(s+day_counter*86400)
            df["elapsed_seconds"]=continuous_time
            df.drop(columns=[time_col],inplace=True)
            print("Time-only dataset converted to continuous elapsed seconds.")
            return df
        except Exception as e:
            print(f"Failed to parse time-only column: {e}.")
        #No datetime detected
        print("No datetime information detected.")
        return df

def intelligent_preprocess(df,numeric_threshold=0.9,cat_threshold=20,text_threshold=0.3):
    #Performs safe numeric conversion, column classification and missing value handling
    #Returns a clean dataset and column metadata
    print("Performing intelligent preprocessing.")
    df=df.copy()
    #Numeric conversion
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            converted=pd.to_numeric(df[col],errors="coerce")
            ratio_numeric=converted.notna().mean()
            if ratio_numeric>=numeric_threshold:
                df[col]=converted
    #Column classification
    col_types={}
    n=len(df)
    for col in df.columns:
        s=df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            col_types[col]="datetime"
            continue
        if pd.api.types.is_numeric_dtype(s):
            unique=s.dropna().nunique()
            col_types[col]="binary" if unique==2 else "numeric"
            continue
        #String and object columns
        unique=s.dropna().nunique()
        ratio_unique=unique/max(n,1) #fraction of rows having unique values
        avg_len=s.dropna().astype(str).str.len().mean()
        if ratio_unique>0.9 and unique>50:#high uniqueness, many distinct values
            col_types[col]="id"
        elif avg_len>30 or ratio_unique>text_threshold:#long strings, many unique values
            col_types[col]="text"
        elif unique<=cat_threshold:#few unique values
            col_types[col]="categorical"
        else:#default
            col_types[col]="text"
    #Missing value handling
    for col,ctype in col_types.items():
        if ctype=="numeric":
            df[col]=df[col].interpolate(limit_direction="both")
        elif ctype=="binary" or ctype=="categorical":
            df[col]=df[col].fillna(df[col].mode().iloc[0])
        elif ctype=="text":
            df[col]=df[col].fillna("")
        elif ctype=="id":
            df[col]=df[col].fillna("UNKNOWN")
    return df,col_types

def remove_outliers(df,z_thresh=4):
    #anything more than 4 standard deviations away from the mean is considered an outlier.
    #replaces extreme values with max or min allowed bounds
    print("Removing outliers.")
    numeric_cols=df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        mean=df[col].mean()
        std=df[col].std()
        z_scores=(df[col]-mean)/std
        df[col]=np.where(
            abs(z_scores)>z_thresh,
            mean+z_thresh*std*np.sign(z_scores),
            df[col]
        )
    return df

def anomaly_detection_engine(data,metadata,window=50):
    print("Detecting anomalies.")
    features=select_anomaly_features(data,metadata)
    if len(features)==0:
        print("No numeric features available for anomaly detection.")
        return
    z_matrix,z_scores=combined_anomaly_detector(data,features,window)
    iso_scores=isolation_forest_detector(data,features)
    scores=pd.DataFrame({
        "statistical":z_scores,
        "isolation":iso_scores
    })
    scores["anomaly_score"]=(
        0.6*scores["statistical"]+0.4*scores["isolation"]
    )#Weighted fusion - statistical engine gets higher weight as it includes long-term deviation and short term spikes
    threshold=scores["anomaly_score"].quantile(0.995)#return 99.5th percentile of the anomaly score distribution
    scores["is_anomaly"]=scores["anomaly_score"]>threshold#boolean
    return scores,z_matrix
    
    
def select_anomaly_features(df,col_types):
    return [
        col for col,ctype in col_types.items() if ctype in ("numeric","binary")
    ]

def combined_anomaly_detector(df,cols,window=50):
    #Calculates largest global z-score for each row; detectes extreme physical deviations
    #Calculates local z-score based on moving averages; detects local short-term anomalies
    #Returns a single anomaly score per row - the largest of local and global z-scores
    scores=pd.DataFrame(index=df.index) #Creates empty dataframe to hold per-column anomaly scores
    for col in cols:
        s=df[col]
        #Global z-score
        mean=s.mean()
        std=s.std()
        global_z=(s-mean)/std
        #Rolling z-score - compared to last 50 values
        rolling_mean=s.rolling(window).mean()
        rolling_std=s.rolling(window).std()
        rolling_z=(s-rolling_mean)/rolling_std
        #Unified score per sensor
        scores[col]=np.nanmax(
            np.vstack([abs(global_z),abs(rolling_z)]),
            axis=0
        )#Compares global and rolling z-scores, ignoring NaNs
    return scores,scores.max(axis=1)#returns largest z-score within each row

def isolation_forest_detector(df,cols):
    #trains an ML model that assigns higher scores to structurally rare data points
    model=IsolationForest(
        n_estimators=200,#build 200 isolation trees
        contamination="auto",#automatically estimate how many anomalies exist
        random_state=42#reproducibility
    )#tree-based algorithm designed for anomaly detection
    X=df[cols].values#extracts numeric columns as a matrix
    model.fit(X)#trains model
    scores=-model.decision_function(X)#returns normality score where negative values are anomalous
    return scores #returns one anomaly score per row

def root_cause_analysis(score_matrix,anomaly_flags,top_k=3):
    #identify root causes for each anomaly, returns dataframe with anomaly index, cause and contributions
    #parameters: anomaly_flags is boolean column True where anomaly detected, top_k is max no. of contributing features
    records=[]#stores one dictionary per anomaly
    anomaly_indices=anomaly_flags[anomaly_flags].index#extract row labels of anomalous rows
    for idx in anomaly_indices:
        row=score_matrix.loc[idx]#extracts per-feature anomaly scores for a given row
        top=row.sort_values(ascending=False).head(top_k)#takes largest 3 anomaly contributions
        records.append({
            "anomaly_index":idx,
            "root_causes":list(top.index),
            "contributions":list(top.values)
        })
    return pd.DataFrame(records)#converts list to dataframe table

def generate_anomaly_report(scores,rca,output_path="anomaly_report.pdf"):
    print("Generating anomaly report.")
    styles=getSampleStyleSheet()#predefined font styles
    story=[]#list containing every element of the report
    #Title
    story.append(Paragraph("Automated Anomaly Detection Report", styles["Title"]))
    story.append(Spacer(1,12))#Vertical spacing below
    #Metadata block
    meta_text = f"""
    <b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
    <b>Total samples:</b> {len(scores)}<br/>
    <b>Total anomalies:</b> {scores['is_anomaly'].sum()}
    """
    story.append(Paragraph(meta_text,styles["Normal"]))
    story.append(Spacer(1, 20))
    #Summary table
    story.append(Paragraph("Anomaly Summary",styles["Heading2"]))
    table_data = [["Index", "Root Causes", "Contributions"]]#creates table header row
    #fill table rows
    for _,row in rca.iterrows():#loop through each anomaly explanation
        table_data.append([
            str(row["anomaly_index"]),
            ", ".join(row["root_causes"]),
            ", ".join(f"{v:.2f}" for v in row["contributions"])#severity contribution z-scores
        ])
    table=Table(table_data,colWidths=[60,300,100])
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
        ('GRID',(0,0),(-1,-1),0.5,colors.grey),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('ALIGN',(0,0),(-1,0),'CENTER'),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE')
    ]))
    story.append(table)#adds table to document
    doc = SimpleDocTemplate(output_path, pagesize=A4)#creates pdf document
    doc.build(story)
    return output_path

file="household_power_consumption.txt"
#file2="test.txt"
data=load_data(file)
validate_data(data)
data,metadata=clean_data(data)
scores,stat_matrix=anomaly_detection_engine(data,metadata)
#print(scores)
print(data)
rca=root_cause_analysis(stat_matrix,scores["is_anomaly"],top_k=3)
#print(rca)
n_file=generate_anomaly_report(scores,rca,"lab_anomaly_report.pdf")