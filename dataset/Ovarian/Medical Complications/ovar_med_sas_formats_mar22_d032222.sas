** Runtime:1966090561.1 Data:/prj/plcoims/maindata/mastdata/monthly/mar22/03.22.22/purple/cancers/ovary/ovar_med.gz **;
proc format;
  ** FORMAT: $buildf **;
  ** FOR VARIABLE: build **;
  ** CHARACTER VARIABLE (formats not searched/validated) **;
  value $buildf
  ;
  ** FORMAT: build_cancersf **;
  ** FOR VARIABLE: build_cancers **;
  value build_cancersf
    1 = "Trial Only"
  ;
  ** FORMAT: build_incidence_cutofff **;
  ** FOR VARIABLE: build_incidence_cutoff **;
  value build_incidence_cutofff
    1 = "Cancers Through 2009"
  ;
  ** FORMAT: $plco_idf **;
  ** FOR VARIABLE: plco_id **;
  ** CHARACTER VARIABLE (formats not searched/validated) **;
  value $plco_idf
  ;
  ** FORMAT: study_yrf **;
  ** FOR VARIABLE: study_yr **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value study_yrf
  ;
  ** FORMAT: canctypef **;
  ** FOR VARIABLE: canctype **;
  value canctypef
    0 = "Non-Cancer"
    1 = "Ovarian Cancer"
    2 = "Peritoneal Cancer"
    3 = "Fallopian Tube Cancer"
    4 = "Ovarian LMP"
  ;
  ** FORMAT: ctype_catof **;
  ** FOR VARIABLE: ctype_cato **;
  value ctype_catof
    1 = "Infection"
    2 = "Direct Surgical"
    3 = "Cardiovascular/Pulmonary"
    9 = "Other"
  ;
  ** FORMAT: ctypeof **;
  ** FOR VARIABLE: ctypeo **;
  value ctypeof
    1 = "Infection (specify)"
    2 = "Fever Requiring Antibiotics"
    3 = "Hemorrhage"
    20 = "Cardiac Arrest"
    21 = "Respiratory arrest"
    22 = "Hospitalization"
    23 = "Pulmonary embolus/emboli"
    24 = "Myocardial infarction"
    25 = "Cardiac arrhythmia"
    26 = "Cerebral vascular accident (CVA)/Stroke"
    27 = "Blood loss requiring transfusion"
    28 = "Deep venous thrombosis (DVT)"
    29 = "Acute/chronic respiratory failure"
    30 = "Hypotension"
    31 = "Congestive heart failure (CHF)"
    32 = "Wound dehiscence"
    33 = "Hypokalemia"
    41 = "Nausea"
    43 = "Electrolyte Imbalance"
    120 = "Urinary"
    121 = "Bleeding"
    122 = "Bowel"
    201 = "Infection and fever"
    202 = "Urinary tract infection and fever"
    400 = "Diarrhea"
    401 = "Small bowel obstruction/partial or complete"
    402 = "Ileus"
    408 = "Bowel injury"
    409 = "Adhesions"
    412 = "Peritonitis"
    413 = "Pneumonia"
    414 = "Urinary tract infection (UTI)"
    415 = "Wound infection"
    420 = "Pain"
  ;
  ** FORMAT: ctypeo_modf **;
  ** FOR VARIABLE: ctypeo_mod **;
  value ctypeo_modf
    1 = "Infection (specify)"
    2 = "Fever Requiring Antibiotics"
    3 = "Hemorrhage"
    20 = "Cardiac Arrest"
    21 = "Respiratory arrest"
    22 = "Hospitalization"
    23 = "Pulmonary embolus/emboli"
    24 = "Myocardial infarction"
    25 = "Cardiac arrhythmia"
    26 = "Cerebral vascular accident (CVA)/Stroke"
    27 = "Blood loss requiring transfusion"
    28 = "Deep venous thrombosis (DVT)"
    29 = "Acute/chronic respiratory failure"
    30 = "Hypotension"
    31 = "Congestive heart failure (CHF)"
    32 = "Wound dehiscence"
    33 = "Hypokalemia"
    41 = "Nausea"
    43 = "Electrolyte Imbalance"
    120 = "Urinary"
    121 = "Bleeding"
    122 = "Bowel"
    201 = "Infection and fever"
    202 = "Urinary tract infection and fever"
    400 = "Diarrhea"
    401 = "Small bowel obstruction/partial or complete"
    402 = "Ileus"
    408 = "Bowel injury"
    409 = "Adhesions"
    412 = "Peritonitis"
    413 = "Pneumonia"
    414 = "Urinary tract infection (UTI)"
    415 = "Wound infection"
    420 = "Pain"
    423 = "Wound Infection"
  ;
  ** FORMAT: med_daysf **;
  ** FOR VARIABLE: med_days **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value med_daysf
  ;
  ** FORMAT: $buildv **;
  ** FOR VARIABLE: build **;
  ** CHARACTER VARIABLE (formats not searched/validated) **;
  value $buildv
  ;
  ** FORMAT: build_cancersv **;
  ** FOR VARIABLE: build_cancers **;
  value build_cancersv
    1 = "(1) Trial Only"
  ;
  ** FORMAT: build_incidence_cutoffv **;
  ** FOR VARIABLE: build_incidence_cutoff **;
  value build_incidence_cutoffv
    1 = "(1) Cancers Through 2009"
  ;
  ** FORMAT: $plco_idv **;
  ** FOR VARIABLE: plco_id **;
  ** CHARACTER VARIABLE (formats not searched/validated) **;
  value $plco_idv
  ;
  ** FORMAT: study_yrv **;
  ** FOR VARIABLE: study_yr **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value study_yrv
  ;
  ** FORMAT: canctypev **;
  ** FOR VARIABLE: canctype **;
  value canctypev
    0 = "(0) Non-Cancer"
    1 = "(1) Ovarian Cancer"
    2 = "(2) Peritoneal Cancer"
    3 = "(3) Fallopian Tube Cancer"
    4 = "(4) Ovarian LMP"
  ;
  ** FORMAT: ctype_catov **;
  ** FOR VARIABLE: ctype_cato **;
  value ctype_catov
    1 = "(1) Infection"
    2 = "(2) Direct Surgical"
    3 = "(3) Cardiovascular/Pulmonary"
    9 = "(9) Other"
  ;
  ** FORMAT: ctypeov **;
  ** FOR VARIABLE: ctypeo **;
  value ctypeov
    1 = "(1) Infection (specify)"
    2 = "(2) Fever Requiring Antibiotics"
    3 = "(3) Hemorrhage"
    20 = "(20) Cardiac Arrest"
    21 = "(21) Respiratory arrest"
    22 = "(22) Hospitalization"
    23 = "(23) Pulmonary embolus/emboli"
    24 = "(24) Myocardial infarction"
    25 = "(25) Cardiac arrhythmia"
    26 = "(26) Cerebral vascular accident (CVA)/Stroke"
    27 = "(27) Blood loss requiring transfusion"
    28 = "(28) Deep venous thrombosis (DVT)"
    29 = "(29) Acute/chronic respiratory failure"
    30 = "(30) Hypotension"
    31 = "(31) Congestive heart failure (CHF)"
    32 = "(32) Wound dehiscence"
    33 = "(33) Hypokalemia"
    41 = "(41) Nausea"
    43 = "(43) Electrolyte Imbalance"
    120 = "(120) Urinary"
    121 = "(121) Bleeding"
    122 = "(122) Bowel"
    201 = "(201) Infection and fever"
    202 = "(202) Urinary tract infection and fever"
    400 = "(400) Diarrhea"
    401 = "(401) Small bowel obstruction/partial or complete"
    402 = "(402) Ileus"
    408 = "(408) Bowel injury"
    409 = "(409) Adhesions"
    412 = "(412) Peritonitis"
    413 = "(413) Pneumonia"
    414 = "(414) Urinary tract infection (UTI)"
    415 = "(415) Wound infection"
    420 = "(420) Pain"
  ;
  ** FORMAT: ctypeo_modv **;
  ** FOR VARIABLE: ctypeo_mod **;
  value ctypeo_modv
    1 = "(1) Infection (specify)"
    2 = "(2) Fever Requiring Antibiotics"
    3 = "(3) Hemorrhage"
    20 = "(20) Cardiac Arrest"
    21 = "(21) Respiratory arrest"
    22 = "(22) Hospitalization"
    23 = "(23) Pulmonary embolus/emboli"
    24 = "(24) Myocardial infarction"
    25 = "(25) Cardiac arrhythmia"
    26 = "(26) Cerebral vascular accident (CVA)/Stroke"
    27 = "(27) Blood loss requiring transfusion"
    28 = "(28) Deep venous thrombosis (DVT)"
    29 = "(29) Acute/chronic respiratory failure"
    30 = "(30) Hypotension"
    31 = "(31) Congestive heart failure (CHF)"
    32 = "(32) Wound dehiscence"
    33 = "(33) Hypokalemia"
    41 = "(41) Nausea"
    43 = "(43) Electrolyte Imbalance"
    120 = "(120) Urinary"
    121 = "(121) Bleeding"
    122 = "(122) Bowel"
    201 = "(201) Infection and fever"
    202 = "(202) Urinary tract infection and fever"
    400 = "(400) Diarrhea"
    401 = "(401) Small bowel obstruction/partial or complete"
    402 = "(402) Ileus"
    408 = "(408) Bowel injury"
    409 = "(409) Adhesions"
    412 = "(412) Peritonitis"
    413 = "(413) Pneumonia"
    414 = "(414) Urinary tract infection (UTI)"
    415 = "(415) Wound infection"
    420 = "(420) Pain"
    423 = "(423) Wound Infection"
  ;
  ** FORMAT: med_daysv **;
  ** FOR VARIABLE: med_days **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value med_daysv
  ;
run;
