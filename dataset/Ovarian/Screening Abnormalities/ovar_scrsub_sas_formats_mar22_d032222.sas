** Runtime:1966090589.0 Data:/prj/plcoims/maindata/mastdata/monthly/mar22/03.22.22/purple/cancers/ovary/ovar_scrsub.gz **;
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
  ** FORMAT: sbcdf **;
  ** FOR VARIABLE: sbcd **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value sbcdf
  ;
  ** FORMAT: $sourcef **;
  ** FOR VARIABLE: source **;
  value $sourcef
    "TVQ" = "TVQ"
    "TVU" = "TVU"
  ;
  ** FORMAT: study_yrf **;
  ** FOR VARIABLE: study_yr **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value study_yrf
  ;
  ** FORMAT: visitf **;
  ** FOR VARIABLE: visit **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value visitf
  ;
  ** FORMAT: cystf **;
  ** FOR VARIABLE: cyst **;
  value cystf
    .M = "Missing"
    1 = "Smooth"
    2 = "Irregularities"
    3 = "Papillarities"
  ;
  ** FORMAT: cystwf **;
  ** FOR VARIABLE: cystw **;
  value cystwf
    .M = "Missing"
    1 = "Thin (< = 3 mm)"
    2 = "Thick (> 3 mm)"
  ;
  ** FORMAT: echof **;
  ** FOR VARIABLE: echo **;
  value echof
    .M = "Missing"
    1 = "Sonolucent"
    2 = "Low"
    3 = "Low with Echogenic Core"
    4 = "Mixed"
    5 = "High"
  ;
  ** FORMAT: maxdif **;
  ** FOR VARIABLE: maxdi **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value maxdif
    .M = "Missing"
  ;
  ** FORMAT: sepstf **;
  ** FOR VARIABLE: sepst **;
  value sepstf
    .M = "Missing"
    0 = "None"
    1 = "Yes, thin (< = 3 mm)"
    2 = "Yes, thick (> 3 mm)"
  ;
  ** FORMAT: $sidef **;
  ** FOR VARIABLE: side **;
  value $sidef
    "L" = "Left"
    "R" = "Right"
  ;
  ** FORMAT: solidf **;
  ** FOR VARIABLE: solid **;
  value solidf
    .M = "Missing"
    0 = "None"
    1 = "Mixed"
    2 = "All Solid"
  ;
  ** FORMAT: volumf **;
  ** FOR VARIABLE: volum **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value volumf
    .M = "Missing"
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
  ** FORMAT: sbcdv **;
  ** FOR VARIABLE: sbcd **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value sbcdv
  ;
  ** FORMAT: $sourcev **;
  ** FOR VARIABLE: source **;
  value $sourcev
    "TVQ" = "(TVQ) TVQ"
    "TVU" = "(TVU) TVU"
  ;
  ** FORMAT: study_yrv **;
  ** FOR VARIABLE: study_yr **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value study_yrv
  ;
  ** FORMAT: visitv **;
  ** FOR VARIABLE: visit **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value visitv
  ;
  ** FORMAT: cystv **;
  ** FOR VARIABLE: cyst **;
  value cystv
    .M = "(.M) Missing"
    1 = "(1) Smooth"
    2 = "(2) Irregularities"
    3 = "(3) Papillarities"
  ;
  ** FORMAT: cystwv **;
  ** FOR VARIABLE: cystw **;
  value cystwv
    .M = "(.M) Missing"
    1 = "(1) Thin (< = 3 mm)"
    2 = "(2) Thick (> 3 mm)"
  ;
  ** FORMAT: echov **;
  ** FOR VARIABLE: echo **;
  value echov
    .M = "(.M) Missing"
    1 = "(1) Sonolucent"
    2 = "(2) Low"
    3 = "(3) Low with Echogenic Core"
    4 = "(4) Mixed"
    5 = "(5) High"
  ;
  ** FORMAT: maxdiv **;
  ** FOR VARIABLE: maxdi **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value maxdiv
    .M = "(.M) Missing"
  ;
  ** FORMAT: sepstv **;
  ** FOR VARIABLE: sepst **;
  value sepstv
    .M = "(.M) Missing"
    0 = "(0) None"
    1 = "(1) Yes, thin (< = 3 mm)"
    2 = "(2) Yes, thick (> 3 mm)"
  ;
  ** FORMAT: $sidev **;
  ** FOR VARIABLE: side **;
  value $sidev
    "L" = "(L) Left"
    "R" = "(R) Right"
  ;
  ** FORMAT: solidv **;
  ** FOR VARIABLE: solid **;
  value solidv
    .M = "(.M) Missing"
    0 = "(0) None"
    1 = "(1) Mixed"
    2 = "(2) All Solid"
  ;
  ** FORMAT: volumv **;
  ** FOR VARIABLE: volum **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value volumv
    .M = "(.M) Missing"
  ;
run;
