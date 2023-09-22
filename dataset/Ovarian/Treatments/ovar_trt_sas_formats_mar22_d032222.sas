** Runtime:1966090581.0 Data:/prj/plcoims/maindata/mastdata/monthly/mar22/03.22.22/purple/cancers/ovary/ovar_trt.gz **;
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
  ** FORMAT: canctypef **;
  ** FOR VARIABLE: canctype **;
  value canctypef
    1 = "Invasive ovarian"
    2 = "Primary peritoneal"
    3 = "Fallopian tube"
    4 = "Ovarian low malignant potential tumor"
    5 = "Peritoneal low malignancy potential tumor"
    6 = "Fallopian tube low malignancy potential tumor"
  ;
  ** FORMAT: trt_daysf **;
  ** FOR VARIABLE: trt_days **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value trt_daysf
  ;
  ** FORMAT: trt_familyof **;
  ** FOR VARIABLE: trt_familyo **;
  value trt_familyof
    1 = "Curative surgery"
    2 = "Chemotherapy"
    3 = "Non-curative treatment"
  ;
  ** FORMAT: trt_numof **;
  ** FOR VARIABLE: trt_numo **;
  value trt_numof
    9 = "Unknown radiation"
    101 = "Bilateral salpingo-oophorectomy with hysterectomy"
    102 = "Omentectomy, with unilateral or bilateral salpingo-oophorectomy, with hysterectomy"
    103 = "Omentectomy, with unilateral or bilateral salpingo-oophorectomy, without hysterectomy"
    105 = "Bilateral salpingo-oophorectomy"
    106 = "Unilateral salpingo-oophorectomy"
    107 = "Unilateral salpingo-oophorectomy, with hysterectomy"
    108 = "Pelvic Exenteration, partial or total"
    109 = "Abdominal/Vaginal hysterectomy"
    110 = "Adhesiolysis"
    111 = "Bowel resection"
    112 = "Lymphadenectomy/lymph node sampling"
    113 = "Omentectomy, complete/NOS"
    114 = "Omentectomy, partial"
    115 = "Resection (specify)"
    116 = "Tumor debulking (cytoreductive surgery)"
    121 = "Abdominal/pelvic organ resection"
    202 = "Cisplatin"
    203 = "Carboplatin"
    206 = "Taxol"
    210 = "CP (cyclophosaphamide/cisplatin)"
    212 = "CAP (cyclophosaphamide/doxorubicin/cisplatin)"
    288 = "Other chemotherapy (specify)"
    510 = "Radiation treatment, NOS"
    520 = "Chemotherapy, NOS"
    530 = "Other treatment, NOS"
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
  ** FORMAT: canctypev **;
  ** FOR VARIABLE: canctype **;
  value canctypev
    1 = "(1) Invasive ovarian"
    2 = "(2) Primary peritoneal"
    3 = "(3) Fallopian tube"
    4 = "(4) Ovarian low malignant potential tumor"
    5 = "(5) Peritoneal low malignancy potential tumor"
    6 = "(6) Fallopian tube low malignancy potential tumor"
  ;
  ** FORMAT: trt_daysv **;
  ** FOR VARIABLE: trt_days **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value trt_daysv
  ;
  ** FORMAT: trt_familyov **;
  ** FOR VARIABLE: trt_familyo **;
  value trt_familyov
    1 = "(1) Curative surgery"
    2 = "(2) Chemotherapy"
    3 = "(3) Non-curative treatment"
  ;
  ** FORMAT: trt_numov **;
  ** FOR VARIABLE: trt_numo **;
  value trt_numov
    9 = "(9) Unknown radiation"
    101 = "(101) Bilateral salpingo-oophorectomy with hysterectomy"
    102 = "(102) Omentectomy, with unilateral or bilateral salpingo-oophorectomy, with hysterectomy"
    103 = "(103) Omentectomy, with unilateral or bilateral salpingo-oophorectomy, without hysterectomy"
    105 = "(105) Bilateral salpingo-oophorectomy"
    106 = "(106) Unilateral salpingo-oophorectomy"
    107 = "(107) Unilateral salpingo-oophorectomy, with hysterectomy"
    108 = "(108) Pelvic Exenteration, partial or total"
    109 = "(109) Abdominal/Vaginal hysterectomy"
    110 = "(110) Adhesiolysis"
    111 = "(111) Bowel resection"
    112 = "(112) Lymphadenectomy/lymph node sampling"
    113 = "(113) Omentectomy, complete/NOS"
    114 = "(114) Omentectomy, partial"
    115 = "(115) Resection (specify)"
    116 = "(116) Tumor debulking (cytoreductive surgery)"
    121 = "(121) Abdominal/pelvic organ resection"
    202 = "(202) Cisplatin"
    203 = "(203) Carboplatin"
    206 = "(206) Taxol"
    210 = "(210) CP (cyclophosaphamide/cisplatin)"
    212 = "(212) CAP (cyclophosaphamide/doxorubicin/cisplatin)"
    288 = "(288) Other chemotherapy (specify)"
    510 = "(510) Radiation treatment, NOS"
    520 = "(520) Chemotherapy, NOS"
    530 = "(530) Other treatment, NOS"
  ;
run;
