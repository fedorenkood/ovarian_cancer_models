** Runtime:1966090563.2 Data:/prj/plcoims/maindata/mastdata/monthly/mar22/03.22.22/purple/cancers/ovary/ovar_proc.gz **;
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
  ** FORMAT: biopf **;
  ** FOR VARIABLE: biop **;
  value biopf
    0 = "No"
    1 = "Yes"
  ;
  ** FORMAT: diag_stagf **;
  ** FOR VARIABLE: diag_stag **;
  value diag_stagf
    1 = "Diagnostic"
    2 = "Staging"
    3 = "Both"
  ;
  ** FORMAT: link_yrf **;
  ** FOR VARIABLE: link_yr **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value link_yrf
    .N = "Not applicable"
  ;
  ** FORMAT: ovar_intstat_cat_linkf **;
  ** FOR VARIABLE: ovar_intstat_cat_link **;
  value ovar_intstat_cat_linkf
    0 = "No cancer"
    1 = "Control with cancer"
    2 = "Never Screened"
    3 = "Post-Screening"
    4 = "Interval"
    5 = "Screen Dx"
  ;
  ** FORMAT: proc_catof **;
  ** FOR VARIABLE: proc_cato **;
  value proc_catof
    1 = "Record Review"
    2 = "Physical Exam/Clinical Evaluation"
    3 = "Repeat Screen"
    4 = "Imaging"
    5 = "Endoscopies & Non-Invasive Scopes"
    6 = "Invasive Scopes, Biopsies, & Surgeries (w/local)"
    7 = "Surgeries (w/general)"
    8 = "Other"
  ;
  ** FORMAT: proc_daysf **;
  ** FOR VARIABLE: proc_days **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value proc_daysf
  ;
  ** FORMAT: proc_numof **;
  ** FOR VARIABLE: proc_numo **;
  value proc_numof
    1 = "CA-125 Blood Test"
    11 = "Barium Enema"
    12 = "Biopsy"
    13 = "Chest Radiograph"
    14 = "Color Doppler"
    15 = "CT Scan - Abdominal"
    16 = "CT Scan - Other"
    17 = "CT Scan - Pelvic"
    18 = "Culdocentesis"
    19 = "Intra-abdominal Washings (Peritoneal or Pelvic)"
    20 = "Intravenous Pyelography (IVP)/ Excretory Urography"
    21 = "Laparotomy"
    23 = "MRI Scan - Abdominal"
    25 = "MRI Scan - Pelvic"
    26 = "Needle Aspiration"
    27 = "Paracentesis"
    28 = "Pelvic Exam by Consulting Physician"
    29 = "Pelvic Exam by Primary Physician"
    30 = "Pelvic Laparoscopy"
    31 = "Transabdominal / Pelvic Ultrasound or Sonogram"
    32 = "Transvaginal Ultrasound"
    33 = "Oophorectomy / Salpingooophorectomy"
    35 = "clinical evaluation"
    36 = "CT scan of abdomen and pelvic combined"
    37 = "CT of chest"
    38 = "hysteroscopy"
    39 = "laparoscopy"
    40 = "lymphadenectomy/lymph node sampling"
    41 = "Omentectomy, complete/NOS"
    42 = "Omentectomy, parial"
    43 = "radiograph, other (specify)"
    44 = "Record Review"
    45 = "Resection (specify)"
    46 = "Sigmoidoscopy/Colonoscopy"
    47 = "Thoracentesis"
    48 = "Transabdominal / pelvic and transvaginal ultrasounds combined"
    49 = "Ultrasound, other (specify)"
    50 = "Abdominal/vaginal hysterectomy"
    60 = "CT - abdomen and thorax"
    61 = "CT - brain/head/skull"
    63 = "CT - chest, abdomen and pelvis"
    64 = "CT - NOS/unknown/unspecified"
    68 = "MRI - NOS/unknown/unspecified"
    77 = "Ultrasound - NOS/unknown/unspecified"
    79 = "Other - PET"
    88 = "MRI - other"
    96 = "Radiograph - other"
    101 = "Other - radionucleotide, Fusion PET/CT"
    104 = "Bx - abdomen/pelvic/peritoneum"
    105 = "Bx - adnexal mass"
    106 = "Bx - bowel/colon"
    107 = "Bx - cyst/cystectomy (ovarian)"
    108 = "Bx - lymph node"
    109 = "Bx - omentum"
    110 = "Radiograph - abdomn"
    111 = "Radiograph - bone"
    112 = "Resection - appendectomy"
    113 = "Resection - colon"
    114 = "Other - ascites"
    115 = "Other - bone scan"
    116 = "Other - tumor debulking"
    117 = "Resection - salpingectomy"
    118 = "Resection - NOS"
    134 = "Bx - endometrium"
    135 = "Other - autopsy"
    136 = "Bx - other"
    888 = "Other Specify"
  ;
  ** FORMAT: scr_linkf **;
  ** FOR VARIABLE: scr_link **;
  value scr_linkf
    0 = "No"
    1 = "Yes"
  ;
  ** FORMAT: bxintentf **;
  ** FOR VARIABLE: bxintent **;
  value bxintentf
    0 = "Not Bx or Rx or Stage"
    1 = "Bx"
    3 = "Bx/Stage"
    4 = "Bx/Rx/Stage"
    6 = "Stage"
    7 = "Rx/Stage"
  ;
  ** FORMAT: pal_link_yearf **;
  ** FOR VARIABLE: pal_link_year **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value pal_link_yearf
    .N = "Not applicable"
  ;
  ** FORMAT: pal_scr_linkf **;
  ** FOR VARIABLE: pal_scr_link **;
  value pal_scr_linkf
    0 = "No, not linked only to OVR screen"
    1 = "Yes, linked only to OVR screen"
  ;
  ** FORMAT: ca125_assayf **;
  ** FOR VARIABLE: ca125_assay **;
  value ca125_assayf
    .N = "Not applicable"
    1 = "Centacour"
    2 = "Abbott"
    8 = "Other (Specify)"
    9 = "Not available"
  ;
  ** FORMAT: ca125resf **;
  ** FOR VARIABLE: ca125res **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ca125resf
    .M = "Missing"
    .N = "Not applicable"
  ;
  ** FORMAT: proc_resf **;
  ** FOR VARIABLE: proc_res **;
  value proc_resf
    .N = "Not applicable"
    .V = "Not captured on this form version"
    1 = "Negative"
    2 = "Abnormal, not suspicious for ovarian cancer"
    3 = "Abnormal, suspicious for ovarian cancer"
    4 = "Abnormal, diagnostic of ovarian cancer (biopsy only)"
    5 = "Abnormal, suspicious for metastasis"
    6 = "Abnormal, confirmed metastasis"
    7 = "Inadequate"
    8 = "Inconclusive"
    9 = "Abnormal, not suspicious for metastasis"
    99 = "Not available"
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
  ** FORMAT: biopv **;
  ** FOR VARIABLE: biop **;
  value biopv
    0 = "(0) No"
    1 = "(1) Yes"
  ;
  ** FORMAT: diag_stagv **;
  ** FOR VARIABLE: diag_stag **;
  value diag_stagv
    1 = "(1) Diagnostic"
    2 = "(2) Staging"
    3 = "(3) Both"
  ;
  ** FORMAT: link_yrv **;
  ** FOR VARIABLE: link_yr **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value link_yrv
    .N = "(.N) Not applicable"
  ;
  ** FORMAT: ovar_intstat_cat_linkv **;
  ** FOR VARIABLE: ovar_intstat_cat_link **;
  value ovar_intstat_cat_linkv
    0 = "(0) No cancer"
    1 = "(1) Control with cancer"
    2 = "(2) Never Screened"
    3 = "(3) Post-Screening"
    4 = "(4) Interval"
    5 = "(5) Screen Dx"
  ;
  ** FORMAT: proc_catov **;
  ** FOR VARIABLE: proc_cato **;
  value proc_catov
    1 = "(1) Record Review"
    2 = "(2) Physical Exam/Clinical Evaluation"
    3 = "(3) Repeat Screen"
    4 = "(4) Imaging"
    5 = "(5) Endoscopies & Non-Invasive Scopes"
    6 = "(6) Invasive Scopes, Biopsies, & Surgeries (w/local)"
    7 = "(7) Surgeries (w/general)"
    8 = "(8) Other"
  ;
  ** FORMAT: proc_daysv **;
  ** FOR VARIABLE: proc_days **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value proc_daysv
  ;
  ** FORMAT: proc_numov **;
  ** FOR VARIABLE: proc_numo **;
  value proc_numov
    1 = "(1) CA-125 Blood Test"
    11 = "(11) Barium Enema"
    12 = "(12) Biopsy"
    13 = "(13) Chest Radiograph"
    14 = "(14) Color Doppler"
    15 = "(15) CT Scan - Abdominal"
    16 = "(16) CT Scan - Other"
    17 = "(17) CT Scan - Pelvic"
    18 = "(18) Culdocentesis"
    19 = "(19) Intra-abdominal Washings (Peritoneal or Pelvic)"
    20 = "(20) Intravenous Pyelography (IVP)/ Excretory Urography"
    21 = "(21) Laparotomy"
    23 = "(23) MRI Scan - Abdominal"
    25 = "(25) MRI Scan - Pelvic"
    26 = "(26) Needle Aspiration"
    27 = "(27) Paracentesis"
    28 = "(28) Pelvic Exam by Consulting Physician"
    29 = "(29) Pelvic Exam by Primary Physician"
    30 = "(30) Pelvic Laparoscopy"
    31 = "(31) Transabdominal / Pelvic Ultrasound or Sonogram"
    32 = "(32) Transvaginal Ultrasound"
    33 = "(33) Oophorectomy / Salpingooophorectomy"
    35 = "(35) clinical evaluation"
    36 = "(36) CT scan of abdomen and pelvic combined"
    37 = "(37) CT of chest"
    38 = "(38) hysteroscopy"
    39 = "(39) laparoscopy"
    40 = "(40) lymphadenectomy/lymph node sampling"
    41 = "(41) Omentectomy, complete/NOS"
    42 = "(42) Omentectomy, parial"
    43 = "(43) radiograph, other (specify)"
    44 = "(44) Record Review"
    45 = "(45) Resection (specify)"
    46 = "(46) Sigmoidoscopy/Colonoscopy"
    47 = "(47) Thoracentesis"
    48 = "(48) Transabdominal / pelvic and transvaginal ultrasounds combined"
    49 = "(49) Ultrasound, other (specify)"
    50 = "(50) Abdominal/vaginal hysterectomy"
    60 = "(60) CT - abdomen and thorax"
    61 = "(61) CT - brain/head/skull"
    63 = "(63) CT - chest, abdomen and pelvis"
    64 = "(64) CT - NOS/unknown/unspecified"
    68 = "(68) MRI - NOS/unknown/unspecified"
    77 = "(77) Ultrasound - NOS/unknown/unspecified"
    79 = "(79) Other - PET"
    88 = "(88) MRI - other"
    96 = "(96) Radiograph - other"
    101 = "(101) Other - radionucleotide, Fusion PET/CT"
    104 = "(104) Bx - abdomen/pelvic/peritoneum"
    105 = "(105) Bx - adnexal mass"
    106 = "(106) Bx - bowel/colon"
    107 = "(107) Bx - cyst/cystectomy (ovarian)"
    108 = "(108) Bx - lymph node"
    109 = "(109) Bx - omentum"
    110 = "(110) Radiograph - abdomn"
    111 = "(111) Radiograph - bone"
    112 = "(112) Resection - appendectomy"
    113 = "(113) Resection - colon"
    114 = "(114) Other - ascites"
    115 = "(115) Other - bone scan"
    116 = "(116) Other - tumor debulking"
    117 = "(117) Resection - salpingectomy"
    118 = "(118) Resection - NOS"
    134 = "(134) Bx - endometrium"
    135 = "(135) Other - autopsy"
    136 = "(136) Bx - other"
    888 = "(888) Other Specify"
  ;
  ** FORMAT: scr_linkv **;
  ** FOR VARIABLE: scr_link **;
  value scr_linkv
    0 = "(0) No"
    1 = "(1) Yes"
  ;
  ** FORMAT: bxintentv **;
  ** FOR VARIABLE: bxintent **;
  value bxintentv
    0 = "(0) Not Bx or Rx or Stage"
    1 = "(1) Bx"
    3 = "(3) Bx/Stage"
    4 = "(4) Bx/Rx/Stage"
    6 = "(6) Stage"
    7 = "(7) Rx/Stage"
  ;
  ** FORMAT: pal_link_yearv **;
  ** FOR VARIABLE: pal_link_year **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value pal_link_yearv
    .N = "(.N) Not applicable"
  ;
  ** FORMAT: pal_scr_linkv **;
  ** FOR VARIABLE: pal_scr_link **;
  value pal_scr_linkv
    0 = "(0) No, not linked only to OVR screen"
    1 = "(1) Yes, linked only to OVR screen"
  ;
  ** FORMAT: ca125_assayv **;
  ** FOR VARIABLE: ca125_assay **;
  value ca125_assayv
    .N = "(.N) Not applicable"
    1 = "(1) Centacour"
    2 = "(2) Abbott"
    8 = "(8) Other (Specify)"
    9 = "(9) Not available"
  ;
  ** FORMAT: ca125resv **;
  ** FOR VARIABLE: ca125res **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ca125resv
    .M = "(.M) Missing"
    .N = "(.N) Not applicable"
  ;
  ** FORMAT: proc_resv **;
  ** FOR VARIABLE: proc_res **;
  value proc_resv
    .N = "(.N) Not applicable"
    .V = "(.V) Not captured on this form version"
    1 = "(1) Negative"
    2 = "(2) Abnormal, not suspicious for ovarian cancer"
    3 = "(3) Abnormal, suspicious for ovarian cancer"
    4 = "(4) Abnormal, diagnostic of ovarian cancer (biopsy only)"
    5 = "(5) Abnormal, suspicious for metastasis"
    6 = "(6) Abnormal, confirmed metastasis"
    7 = "(7) Inadequate"
    8 = "(8) Inconclusive"
    9 = "(9) Abnormal, not suspicious for metastasis"
    99 = "(99) Not available"
  ;
run;
