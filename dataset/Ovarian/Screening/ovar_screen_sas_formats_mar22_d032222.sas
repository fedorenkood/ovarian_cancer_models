** Runtime:1966090586.4 Data:/prj/plcoims/maindata/mastdata/monthly/mar22/03.22.22/purple/cancers/ovary/ovar_screen.gz **;
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
  ** FORMAT: ovar_daysf **;
  ** FOR VARIABLE: ovar_days **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ovar_daysf
    .F = "No form"
  ;
  ** FORMAT: ovar_resultf **;
  ** FOR VARIABLE: ovar_result **;
  value ovar_resultf
    1 = "Negative"
    2 = "Abnormal, suspicious"
    3 = "Abnormal, non-suspicious"
    4 = "Inadequate screen"
    8 = "Not done, expected"
    9 = "Not done, not expected"
  ;
  ** FORMAT: ca125_daysf **;
  ** FOR VARIABLE: ca125_days **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ca125_daysf
    .F = "No form"
  ;
  ** FORMAT: ca125_levelf **;
  ** FOR VARIABLE: ca125_level **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ca125_levelf
    .F = "No form"
  ;
  ** FORMAT: ca125_resultf **;
  ** FOR VARIABLE: ca125_result **;
  value ca125_resultf
    1 = "Negative"
    2 = "Abnormal, Suspicious"
    4 = "Inadequate"
    8 = "Not done, expected"
    9 = "Not done, not expected"
  ;
  ** FORMAT: ca125_srcf **;
  ** FOR VARIABLE: ca125_src **;
  value ca125_srcf
    .F = "No form"
    1 = "Version 1"
    2 = "Version 2"
  ;
  ** FORMAT: ca125i_assess_daysf **;
  ** FOR VARIABLE: ca125i_assess_days **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ca125i_assess_daysf
    .M = "Missing"
  ;
  ** FORMAT: ca125ii_assess_daysf **;
  ** FOR VARIABLE: ca125ii_assess_days **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ca125ii_assess_daysf
    .M = "Missing"
  ;
  ** FORMAT: ca125ii_srcf **;
  ** FOR VARIABLE: ca125ii_src **;
  value ca125ii_srcf
    .F = "No form"
    1 = "PCR"
    2 = "UCLA"
  ;
  ** FORMAT: tvu_daysf **;
  ** FOR VARIABLE: tvu_days **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value tvu_daysf
    .F = "No form"
    .N = "Not Applicable"
  ;
  ** FORMAT: tvu_resultf **;
  ** FOR VARIABLE: tvu_result **;
  value tvu_resultf
    1 = "Negative"
    2 = "Abnormal, Suspicious"
    3 = "Abnormal, Non-Suspicious"
    4 = "Inadequate"
    8 = "Not done, expected"
    9 = "Not done, not expected"
  ;
  ** FORMAT: tvudays_pvis1f **;
  ** FOR VARIABLE: tvudays_pvis1-3 **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value tvudays_pvis1f
    .F = "No Form"
    .N = "Not applicable"
  ;
  ** FORMAT: tvures_pvis1f **;
  ** FOR VARIABLE: tvures_pvis1-3 **;
  value tvures_pvis1f
    .F = "No Form"
    .N = "Not Applicable"
    1 = "NG"
    2 = "AS"
    3 = "IN"
    4 = "AN"
  ;
  ** FORMAT: tvures_qvis1f **;
  ** FOR VARIABLE: tvures_qvis1-3 **;
  value tvures_qvis1f
    .F = "No Form"
    .N = "Not Applicable"
    1 = "NG"
    2 = "AS"
    3 = "IN"
    4 = "AN"
  ;
  ** FORMAT: tvu_assess_days_qf **;
  ** FOR VARIABLE: tvu_assess_days_q **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value tvu_assess_days_qf
    .F = "No Form"
  ;
  ** FORMAT: tvu_reff **;
  ** FOR VARIABLE: tvu_ref **;
  value tvu_reff
    .M = "Missing"
    1 = "Significant Abnormality, Referral"
    2 = "Moderate Abnormality, Referral"
    3 = "Slight Variation from Normal, No Referral"
    4 = "Normal / Result Not Available, No Referral"
  ;
  ** FORMAT: numcystf **;
  ** FOR VARIABLE: numcyst **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value numcystf
  ;
  ** FORMAT: ovary_diamf **;
  ** FOR VARIABLE: ovary_diam **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ovary_diamf
    .F = "No Form"
    .M = "Missing"
    .N = "Not visualized"
  ;
  ** FORMAT: ovary_volf **;
  ** FOR VARIABLE: ovary_vol **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ovary_volf
    .F = "No Form"
    .M = "Missing"
    .N = "Not visualized"
  ;
  ** FORMAT: ovcyst_diamf **;
  ** FOR VARIABLE: ovcyst_diam **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ovcyst_diamf
    .F = "No Form"
    .M = "Missing"
    .N = "Not visualized"
  ;
  ** FORMAT: ovcyst_morphf **;
  ** FOR VARIABLE: ovcyst_morph **;
  value ovcyst_morphf
    0 = "Not visualized"
    1 = "No solid area and outline smooth"
    2 = " Mixed solid area or outline irregular or outline papillaries"
    3 = "Solid area"
  ;
  ** FORMAT: ovcyst_outlinef **;
  ** FOR VARIABLE: ovcyst_outline **;
  value ovcyst_outlinef
    .F = "No Form"
    .M = "Missing"
    .N = "Not visualized"
    1 = "Smooth"
    2 = "Irregularities"
    3 = "Papillarities"
  ;
  ** FORMAT: ovcyst_solidf **;
  ** FOR VARIABLE: ovcyst_solid **;
  value ovcyst_solidf
    .F = "No Form"
    .M = "Missing"
    .N = "Not visualized"
    0 = "None"
    1 = "Mixed"
    2 = "All Solid"
  ;
  ** FORMAT: ovcyst_sumf **;
  ** FOR VARIABLE: ovcyst_sum **;
  value ovcyst_sumf
    0 = "Not visualized"
    1 = "< 3cm, no solid area and smooth outline"
    2 = "< 3cm, mixed solid area or irregular/papillary outline"
    3 = "< 3cm, solid area"
    4 = "3 -< 6cm, no solid area and smooth outline"
    5 = "3 -< 6cm, mixed solid area or irregular/papillary outline"
    6 = "3 -< 6cm, solid area"
    7 = "6+ cm, no solid area and smooth outline"
    8 = "6+ cm, mixed solid area or irregular/papillary outline"
    9 = "6+ cm, solid area"
  ;
  ** FORMAT: ovcyst_volf **;
  ** FOR VARIABLE: ovcyst_vol **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ovcyst_volf
    .F = "No Form"
    .M = "Missing"
    .N = "Not visualized"
  ;
  ** FORMAT: visbothf **;
  ** FOR VARIABLE: visboth **;
  value visbothf
    0 = "No"
    1 = "Yes"
  ;
  ** FORMAT: viseithf **;
  ** FOR VARIABLE: viseith **;
  value viseithf
    0 = "No"
    1 = "Yes"
  ;
  ** FORMAT: vislf **;
  ** FOR VARIABLE: visl **;
  value vislf
    0 = "No"
    1 = "Yes"
  ;
  ** FORMAT: visrf **;
  ** FOR VARIABLE: visr **;
  value visrf
    0 = "No"
    1 = "Yes"
  ;
  ** FORMAT: detl_pf **;
  ** FOR VARIABLE: detl_p **;
  value detl_pf
    .F = "No form"
    .M = "Missing"
    0 = "No"
    1 = "Yes"
  ;
  ** FORMAT: detl_qf **;
  ** FOR VARIABLE: detl_q **;
  value detl_qf
    .F = "No form"
    .M = "Missing"
    0 = "No"
    1 = "Yes"
  ;
  ** FORMAT: detr_pf **;
  ** FOR VARIABLE: detr_p **;
  value detr_pf
    .F = "No form"
    .M = "Missing"
    0 = "No"
    1 = "Yes"
  ;
  ** FORMAT: detr_qf **;
  ** FOR VARIABLE: detr_q **;
  value detr_qf
    .F = "No form"
    .M = "Missing"
    0 = "No"
    1 = "Yes"
  ;
  ** FORMAT: lantero_pf **;
  ** FOR VARIABLE: lantero_p **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value lantero_pf
    .M = "Missing"
  ;
  ** FORMAT: lantero_qf **;
  ** FOR VARIABLE: lantero_q **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value lantero_qf
    .M = "Missing"
  ;
  ** FORMAT: llong_pf **;
  ** FOR VARIABLE: llong_p **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value llong_pf
    .M = "Missing"
  ;
  ** FORMAT: llong_qf **;
  ** FOR VARIABLE: llong_q **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value llong_qf
    .M = "Missing"
  ;
  ** FORMAT: ltran_pf **;
  ** FOR VARIABLE: ltran_p **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ltran_pf
    .M = "Missing"
  ;
  ** FORMAT: ltran_qf **;
  ** FOR VARIABLE: ltran_q **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ltran_qf
    .M = "Missing"
  ;
  ** FORMAT: lvol_pf **;
  ** FOR VARIABLE: lvol_p **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value lvol_pf
    . = "Missing or not applicable"
  ;
  ** FORMAT: lvol_qf **;
  ** FOR VARIABLE: lvol_q **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value lvol_qf
    . = "Missing or not applicable"
  ;
  ** FORMAT: rantero_pf **;
  ** FOR VARIABLE: rantero_p **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value rantero_pf
    .M = "Missing"
  ;
  ** FORMAT: rantero_qf **;
  ** FOR VARIABLE: rantero_q **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value rantero_qf
    .M = "Missing"
  ;
  ** FORMAT: rlong_pf **;
  ** FOR VARIABLE: rlong_p **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value rlong_pf
    .M = "Missing"
  ;
  ** FORMAT: rlong_qf **;
  ** FOR VARIABLE: rlong_q **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value rlong_qf
    .M = "Missing"
  ;
  ** FORMAT: rtran_pf **;
  ** FOR VARIABLE: rtran_p **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value rtran_pf
    .M = "Missing"
  ;
  ** FORMAT: rtran_qf **;
  ** FOR VARIABLE: rtran_q **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value rtran_qf
    .M = "Missing"
  ;
  ** FORMAT: rvol_pf **;
  ** FOR VARIABLE: rvol_p **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value rvol_pf
    . = "Missing or not applicable"
  ;
  ** FORMAT: rvol_qf **;
  ** FOR VARIABLE: rvol_q **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value rvol_qf
    . = "Missing or not applicable"
  ;
  ** FORMAT: numcystlf **;
  ** FOR VARIABLE: numcystl **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value numcystlf
  ;
  ** FORMAT: numcystrf **;
  ** FOR VARIABLE: numcystr **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value numcystrf
  ;
  ** FORMAT: ovary_diamlf **;
  ** FOR VARIABLE: ovary_diaml **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ovary_diamlf
    .F = "No Form"
    .M = "Missing"
    .N = "Not visualized"
  ;
  ** FORMAT: ovary_diamrf **;
  ** FOR VARIABLE: ovary_diamr **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ovary_diamrf
    .F = "No Form"
    .M = "Missing"
    .N = "Not visualized"
  ;
  ** FORMAT: ovary_vollf **;
  ** FOR VARIABLE: ovary_voll **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ovary_vollf
    .F = "No Form"
    .M = "Missing"
    .N = "Not visualized"
  ;
  ** FORMAT: ovary_volrf **;
  ** FOR VARIABLE: ovary_volr **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ovary_volrf
    .F = "No Form"
    .M = "Missing"
    .N = "Not visualized"
  ;
  ** FORMAT: ovcyst_diamlf **;
  ** FOR VARIABLE: ovcyst_diaml **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ovcyst_diamlf
    .F = "No Form"
    .M = "Missing"
    .N = "Not visualized"
  ;
  ** FORMAT: ovcyst_diamrf **;
  ** FOR VARIABLE: ovcyst_diamr **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ovcyst_diamrf
    .F = "No Form"
    .M = "Missing"
    .N = "Not visualized"
  ;
  ** FORMAT: ovcyst_morphlf **;
  ** FOR VARIABLE: ovcyst_morphl **;
  value ovcyst_morphlf
    0 = "Not visualized"
    1 = "No solid area and outline smooth"
    2 = " Mixed solid area or outline irregular or outline papillaries"
    3 = "Solid area"
  ;
  ** FORMAT: ovcyst_morphrf **;
  ** FOR VARIABLE: ovcyst_morphr **;
  value ovcyst_morphrf
    0 = "Not visualized"
    1 = "No solid area and outline smooth"
    2 = " Mixed solid area or outline irregular or outline papillaries"
    3 = "Solid area"
  ;
  ** FORMAT: ovcyst_outlinelf **;
  ** FOR VARIABLE: ovcyst_outlinel **;
  value ovcyst_outlinelf
    .F = "No Form"
    .M = "Missing"
    .N = "Not visualized"
    1 = "Smooth"
    2 = "Irregularities"
    3 = "Papillarities"
  ;
  ** FORMAT: ovcyst_outlinerf **;
  ** FOR VARIABLE: ovcyst_outliner **;
  value ovcyst_outlinerf
    .F = "No Form"
    .M = "Missing"
    .N = "Not visualized"
    1 = "Smooth"
    2 = "Irregularities"
    3 = "Papillarities"
  ;
  ** FORMAT: ovcyst_solidlf **;
  ** FOR VARIABLE: ovcyst_solidl **;
  value ovcyst_solidlf
    .F = "No Form"
    .M = "Missing"
    .N = "Not visualized"
    0 = "None"
    1 = "Mixed"
    2 = "All Solid"
  ;
  ** FORMAT: ovcyst_solidrf **;
  ** FOR VARIABLE: ovcyst_solidr **;
  value ovcyst_solidrf
    .F = "No Form"
    .M = "Missing"
    .N = "Not visualized"
    0 = "None"
    1 = "Mixed"
    2 = "All Solid"
  ;
  ** FORMAT: ovcyst_sumlf **;
  ** FOR VARIABLE: ovcyst_suml **;
  value ovcyst_sumlf
    0 = "Not visualized"
    1 = "< 3cm, no solid area and smooth outline"
    2 = "< 3cm, mixed solid area or irregular/papillary outline"
    3 = "< 3cm, solid area"
    4 = "3 -< 6cm, no solid area and smooth outline"
    5 = "3 -< 6cm, mixed solid area or irregular/papillary outline"
    6 = "3 -< 6cm, solid area"
    7 = "6+ cm, no solid area and smooth outline"
    8 = "6+ cm, mixed solid area or irregular/papillary outline"
    9 = "6+ cm, solid area"
  ;
  ** FORMAT: ovcyst_sumrf **;
  ** FOR VARIABLE: ovcyst_sumr **;
  value ovcyst_sumrf
    0 = "Not visualized"
    1 = "< 3cm, no solid area and smooth outline"
    2 = "< 3cm, mixed solid area or irregular/papillary outline"
    3 = "< 3cm, solid area"
    4 = "3 -< 6cm, no solid area and smooth outline"
    5 = "3 -< 6cm, mixed solid area or irregular/papillary outline"
    6 = "3 -< 6cm, solid area"
    7 = "6+ cm, no solid area and smooth outline"
    8 = "6+ cm, mixed solid area or irregular/papillary outline"
    9 = "6+ cm, solid area"
  ;
  ** FORMAT: ovcyst_vollf **;
  ** FOR VARIABLE: ovcyst_voll **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ovcyst_vollf
    .F = "No Form"
    .M = "Missing"
    .N = "Not visualized"
  ;
  ** FORMAT: ovcyst_volrf **;
  ** FOR VARIABLE: ovcyst_volr **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ovcyst_volrf
    .F = "No Form"
    .M = "Missing"
    .N = "Not visualized"
  ;
  ** FORMAT: $examineridf **;
  ** FOR VARIABLE: examinerid **;
  ** CHARACTER VARIABLE (formats not searched/validated) **;
  value $examineridf
  ;
  ** FORMAT: inad_bowf **;
  ** FOR VARIABLE: inad_bow **;
  value inad_bowf
    .M = "Not Marked"
    1 = "Yes"
  ;
  ** FORMAT: inad_disf **;
  ** FOR VARIABLE: inad_dis **;
  value inad_disf
    .M = "Not Marked"
    1 = "Yes"
  ;
  ** FORMAT: inad_malf **;
  ** FOR VARIABLE: inad_mal **;
  value inad_malf
    .M = "Not Marked"
    1 = "Yes"
  ;
  ** FORMAT: inad_othf **;
  ** FOR VARIABLE: inad_oth **;
  value inad_othf
    .M = "Not Marked"
    1 = "Yes"
  ;
  ** FORMAT: inad_probef **;
  ** FOR VARIABLE: inad_probe **;
  value inad_probef
    .M = "Not Marked"
    1 = "Yes"
  ;
  ** FORMAT: inad_reff **;
  ** FOR VARIABLE: inad_ref **;
  value inad_reff
    .M = "Not Marked"
    1 = "Yes"
  ;
  ** FORMAT: medcompf **;
  ** FOR VARIABLE: medcomp **;
  value medcompf
    .F = "No form"
    .M = "Missing"
    0 = "No"
    1 = "Yes"
  ;
  ** FORMAT: qamethodf **;
  ** FOR VARIABLE: qamethod **;
  value qamethodf
    .F = "No form"
    .M = "Missing"
    1 = "Repeat examination"
    2 = "Observation of examination"
    3 = "Review of files"
  ;
  ** FORMAT: $examinerid_pvis1f **;
  ** FOR VARIABLE: examinerid_pvis1-3 **;
  ** CHARACTER VARIABLE (formats not searched/validated) **;
  value $examinerid_pvis1f
  ;
  ** FORMAT: $examinerid_qvis1f **;
  ** FOR VARIABLE: examinerid_qvis1-3 **;
  ** CHARACTER VARIABLE (formats not searched/validated) **;
  value $examinerid_qvis1f
  ;
  ** FORMAT: inad_bow_p1f **;
  ** FOR VARIABLE: inad_bow_p1-3 **;
  value inad_bow_p1f
    .F = "No Form"
    .N = "Not Applicable"
    1 = "Yes"
  ;
  ** FORMAT: inad_bow_q1f **;
  ** FOR VARIABLE: inad_bow_q1-3 **;
  value inad_bow_q1f
    .F = "No Form"
    .N = "Not Applicable"
    1 = "Yes"
  ;
  ** FORMAT: inad_dis_p1f **;
  ** FOR VARIABLE: inad_dis_p1-3 **;
  value inad_dis_p1f
    .F = "No Form"
    .N = "Not Applicable"
    1 = "Yes"
  ;
  ** FORMAT: inad_dis_q1f **;
  ** FOR VARIABLE: inad_dis_q1-3 **;
  value inad_dis_q1f
    .F = "No Form"
    .N = "Not Applicable"
    1 = "Yes"
  ;
  ** FORMAT: inad_mal_p1f **;
  ** FOR VARIABLE: inad_mal_p1-3 **;
  value inad_mal_p1f
    .F = "No Form"
    .N = "Not Applicable"
    1 = "Yes"
  ;
  ** FORMAT: inad_mal_q1f **;
  ** FOR VARIABLE: inad_mal_q1-3 **;
  value inad_mal_q1f
    .F = "No Form"
    .N = "Not Applicable"
    1 = "Yes"
  ;
  ** FORMAT: inad_oth_p1f **;
  ** FOR VARIABLE: inad_oth_p1-3 **;
  value inad_oth_p1f
    .F = "No Form"
    .N = "Not Applicable"
    1 = "Yes"
  ;
  ** FORMAT: inad_oth_q1f **;
  ** FOR VARIABLE: inad_oth_q1-3 **;
  value inad_oth_q1f
    .F = "No Form"
    .N = "Not Applicable"
    1 = "Yes"
  ;
  ** FORMAT: inad_probe_p1f **;
  ** FOR VARIABLE: inad_probe_p1-3 **;
  value inad_probe_p1f
    .F = "No Form"
    .N = "Not Applicable"
    1 = "Yes"
  ;
  ** FORMAT: inad_probe_q1f **;
  ** FOR VARIABLE: inad_probe_q1-3 **;
  value inad_probe_q1f
    .F = "No Form"
    .N = "Not Applicable"
    1 = "Yes"
  ;
  ** FORMAT: inad_ref_p1f **;
  ** FOR VARIABLE: inad_ref_p1-3 **;
  value inad_ref_p1f
    .F = "No Form"
    .N = "Not Applicable"
    1 = "Yes"
  ;
  ** FORMAT: inad_ref_q1f **;
  ** FOR VARIABLE: inad_ref_q1-3 **;
  value inad_ref_q1f
    .F = "No Form"
    .N = "Not Applicable"
    1 = "Yes"
  ;
  ** FORMAT: phyconsf **;
  ** FOR VARIABLE: phycons **;
  value phyconsf
    .F = "No form"
    .M = "Missing"
    0 = "No"
    1 = "Yes"
  ;
  ** FORMAT: physidf **;
  ** FOR VARIABLE: physid **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value physidf
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
  ** FORMAT: study_yrv **;
  ** FOR VARIABLE: study_yr **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value study_yrv
  ;
  ** FORMAT: ovar_daysv **;
  ** FOR VARIABLE: ovar_days **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ovar_daysv
    .F = "(.F) No form"
  ;
  ** FORMAT: ovar_resultv **;
  ** FOR VARIABLE: ovar_result **;
  value ovar_resultv
    1 = "(1) Negative"
    2 = "(2) Abnormal, suspicious"
    3 = "(3) Abnormal, non-suspicious"
    4 = "(4) Inadequate screen"
    8 = "(8) Not done, expected"
    9 = "(9) Not done, not expected"
  ;
  ** FORMAT: ca125_daysv **;
  ** FOR VARIABLE: ca125_days **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ca125_daysv
    .F = "(.F) No form"
  ;
  ** FORMAT: ca125_levelv **;
  ** FOR VARIABLE: ca125_level **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ca125_levelv
    .F = "(.F) No form"
  ;
  ** FORMAT: ca125_resultv **;
  ** FOR VARIABLE: ca125_result **;
  value ca125_resultv
    1 = "(1) Negative"
    2 = "(2) Abnormal, Suspicious"
    4 = "(4) Inadequate"
    8 = "(8) Not done, expected"
    9 = "(9) Not done, not expected"
  ;
  ** FORMAT: ca125_srcv **;
  ** FOR VARIABLE: ca125_src **;
  value ca125_srcv
    .F = "(.F) No form"
    1 = "(1) Version 1"
    2 = "(2) Version 2"
  ;
  ** FORMAT: ca125i_assess_daysv **;
  ** FOR VARIABLE: ca125i_assess_days **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ca125i_assess_daysv
    .M = "(.M) Missing"
  ;
  ** FORMAT: ca125ii_assess_daysv **;
  ** FOR VARIABLE: ca125ii_assess_days **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ca125ii_assess_daysv
    .M = "(.M) Missing"
  ;
  ** FORMAT: ca125ii_srcv **;
  ** FOR VARIABLE: ca125ii_src **;
  value ca125ii_srcv
    .F = "(.F) No form"
    1 = "(1) PCR"
    2 = "(2) UCLA"
  ;
  ** FORMAT: tvu_daysv **;
  ** FOR VARIABLE: tvu_days **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value tvu_daysv
    .F = "(.F) No form"
    .N = "(.N) Not Applicable"
  ;
  ** FORMAT: tvu_resultv **;
  ** FOR VARIABLE: tvu_result **;
  value tvu_resultv
    1 = "(1) Negative"
    2 = "(2) Abnormal, Suspicious"
    3 = "(3) Abnormal, Non-Suspicious"
    4 = "(4) Inadequate"
    8 = "(8) Not done, expected"
    9 = "(9) Not done, not expected"
  ;
  ** FORMAT: tvudays_pvis1v **;
  ** FOR VARIABLE: tvudays_pvis1-3 **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value tvudays_pvis1v
    .F = "(.F) No Form"
    .N = "(.N) Not applicable"
  ;
  ** FORMAT: tvures_pvis1v **;
  ** FOR VARIABLE: tvures_pvis1-3 **;
  value tvures_pvis1v
    .F = "(.F) No Form"
    .N = "(.N) Not Applicable"
    1 = "(1) NG"
    2 = "(2) AS"
    3 = "(3) IN"
    4 = "(4) AN"
  ;
  ** FORMAT: tvures_qvis1v **;
  ** FOR VARIABLE: tvures_qvis1-3 **;
  value tvures_qvis1v
    .F = "(.F) No Form"
    .N = "(.N) Not Applicable"
    1 = "(1) NG"
    2 = "(2) AS"
    3 = "(3) IN"
    4 = "(4) AN"
  ;
  ** FORMAT: tvu_assess_days_qv **;
  ** FOR VARIABLE: tvu_assess_days_q **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value tvu_assess_days_qv
    .F = "(.F) No Form"
  ;
  ** FORMAT: tvu_refv **;
  ** FOR VARIABLE: tvu_ref **;
  value tvu_refv
    .M = "(.M) Missing"
    1 = "(1) Significant Abnormality, Referral"
    2 = "(2) Moderate Abnormality, Referral"
    3 = "(3) Slight Variation from Normal, No Referral"
    4 = "(4) Normal / Result Not Available, No Referral"
  ;
  ** FORMAT: numcystv **;
  ** FOR VARIABLE: numcyst **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value numcystv
  ;
  ** FORMAT: ovary_diamv **;
  ** FOR VARIABLE: ovary_diam **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ovary_diamv
    .F = "(.F) No Form"
    .M = "(.M) Missing"
    .N = "(.N) Not visualized"
  ;
  ** FORMAT: ovary_volv **;
  ** FOR VARIABLE: ovary_vol **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ovary_volv
    .F = "(.F) No Form"
    .M = "(.M) Missing"
    .N = "(.N) Not visualized"
  ;
  ** FORMAT: ovcyst_diamv **;
  ** FOR VARIABLE: ovcyst_diam **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ovcyst_diamv
    .F = "(.F) No Form"
    .M = "(.M) Missing"
    .N = "(.N) Not visualized"
  ;
  ** FORMAT: ovcyst_morphv **;
  ** FOR VARIABLE: ovcyst_morph **;
  value ovcyst_morphv
    0 = "(0) Not visualized"
    1 = "(1) No solid area and outline smooth"
    2 = "(2) Mixed solid area or outline irregular or outline papillaries"
    3 = "(3) Solid area"
  ;
  ** FORMAT: ovcyst_outlinev **;
  ** FOR VARIABLE: ovcyst_outline **;
  value ovcyst_outlinev
    .F = "(.F) No Form"
    .M = "(.M) Missing"
    .N = "(.N) Not visualized"
    1 = "(1) Smooth"
    2 = "(2) Irregularities"
    3 = "(3) Papillarities"
  ;
  ** FORMAT: ovcyst_solidv **;
  ** FOR VARIABLE: ovcyst_solid **;
  value ovcyst_solidv
    .F = "(.F) No Form"
    .M = "(.M) Missing"
    .N = "(.N) Not visualized"
    0 = "(0) None"
    1 = "(1) Mixed"
    2 = "(2) All Solid"
  ;
  ** FORMAT: ovcyst_sumv **;
  ** FOR VARIABLE: ovcyst_sum **;
  value ovcyst_sumv
    0 = "(0) Not visualized"
    1 = "(1) < 3cm, no solid area and smooth outline"
    2 = "(2) < 3cm, mixed solid area or irregular/papillary outline"
    3 = "(3) < 3cm, solid area"
    4 = "(4) 3 -< 6cm, no solid area and smooth outline"
    5 = "(5) 3 -< 6cm, mixed solid area or irregular/papillary outline"
    6 = "(6) 3 -< 6cm, solid area"
    7 = "(7) 6+ cm, no solid area and smooth outline"
    8 = "(8) 6+ cm, mixed solid area or irregular/papillary outline"
    9 = "(9) 6+ cm, solid area"
  ;
  ** FORMAT: ovcyst_volv **;
  ** FOR VARIABLE: ovcyst_vol **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ovcyst_volv
    .F = "(.F) No Form"
    .M = "(.M) Missing"
    .N = "(.N) Not visualized"
  ;
  ** FORMAT: visbothv **;
  ** FOR VARIABLE: visboth **;
  value visbothv
    0 = "(0) No"
    1 = "(1) Yes"
  ;
  ** FORMAT: viseithv **;
  ** FOR VARIABLE: viseith **;
  value viseithv
    0 = "(0) No"
    1 = "(1) Yes"
  ;
  ** FORMAT: vislv **;
  ** FOR VARIABLE: visl **;
  value vislv
    0 = "(0) No"
    1 = "(1) Yes"
  ;
  ** FORMAT: visrv **;
  ** FOR VARIABLE: visr **;
  value visrv
    0 = "(0) No"
    1 = "(1) Yes"
  ;
  ** FORMAT: detl_pv **;
  ** FOR VARIABLE: detl_p **;
  value detl_pv
    .F = "(.F) No form"
    .M = "(.M) Missing"
    0 = "(0) No"
    1 = "(1) Yes"
  ;
  ** FORMAT: detl_qv **;
  ** FOR VARIABLE: detl_q **;
  value detl_qv
    .F = "(.F) No form"
    .M = "(.M) Missing"
    0 = "(0) No"
    1 = "(1) Yes"
  ;
  ** FORMAT: detr_pv **;
  ** FOR VARIABLE: detr_p **;
  value detr_pv
    .F = "(.F) No form"
    .M = "(.M) Missing"
    0 = "(0) No"
    1 = "(1) Yes"
  ;
  ** FORMAT: detr_qv **;
  ** FOR VARIABLE: detr_q **;
  value detr_qv
    .F = "(.F) No form"
    .M = "(.M) Missing"
    0 = "(0) No"
    1 = "(1) Yes"
  ;
  ** FORMAT: lantero_pv **;
  ** FOR VARIABLE: lantero_p **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value lantero_pv
    .M = "(.M) Missing"
  ;
  ** FORMAT: lantero_qv **;
  ** FOR VARIABLE: lantero_q **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value lantero_qv
    .M = "(.M) Missing"
  ;
  ** FORMAT: llong_pv **;
  ** FOR VARIABLE: llong_p **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value llong_pv
    .M = "(.M) Missing"
  ;
  ** FORMAT: llong_qv **;
  ** FOR VARIABLE: llong_q **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value llong_qv
    .M = "(.M) Missing"
  ;
  ** FORMAT: ltran_pv **;
  ** FOR VARIABLE: ltran_p **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ltran_pv
    .M = "(.M) Missing"
  ;
  ** FORMAT: ltran_qv **;
  ** FOR VARIABLE: ltran_q **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ltran_qv
    .M = "(.M) Missing"
  ;
  ** FORMAT: lvol_pv **;
  ** FOR VARIABLE: lvol_p **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value lvol_pv
    . = "(.) Missing or not applicable"
  ;
  ** FORMAT: lvol_qv **;
  ** FOR VARIABLE: lvol_q **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value lvol_qv
    . = "(.) Missing or not applicable"
  ;
  ** FORMAT: rantero_pv **;
  ** FOR VARIABLE: rantero_p **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value rantero_pv
    .M = "(.M) Missing"
  ;
  ** FORMAT: rantero_qv **;
  ** FOR VARIABLE: rantero_q **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value rantero_qv
    .M = "(.M) Missing"
  ;
  ** FORMAT: rlong_pv **;
  ** FOR VARIABLE: rlong_p **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value rlong_pv
    .M = "(.M) Missing"
  ;
  ** FORMAT: rlong_qv **;
  ** FOR VARIABLE: rlong_q **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value rlong_qv
    .M = "(.M) Missing"
  ;
  ** FORMAT: rtran_pv **;
  ** FOR VARIABLE: rtran_p **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value rtran_pv
    .M = "(.M) Missing"
  ;
  ** FORMAT: rtran_qv **;
  ** FOR VARIABLE: rtran_q **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value rtran_qv
    .M = "(.M) Missing"
  ;
  ** FORMAT: rvol_pv **;
  ** FOR VARIABLE: rvol_p **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value rvol_pv
    . = "(.) Missing or not applicable"
  ;
  ** FORMAT: rvol_qv **;
  ** FOR VARIABLE: rvol_q **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value rvol_qv
    . = "(.) Missing or not applicable"
  ;
  ** FORMAT: numcystlv **;
  ** FOR VARIABLE: numcystl **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value numcystlv
  ;
  ** FORMAT: numcystrv **;
  ** FOR VARIABLE: numcystr **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value numcystrv
  ;
  ** FORMAT: ovary_diamlv **;
  ** FOR VARIABLE: ovary_diaml **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ovary_diamlv
    .F = "(.F) No Form"
    .M = "(.M) Missing"
    .N = "(.N) Not visualized"
  ;
  ** FORMAT: ovary_diamrv **;
  ** FOR VARIABLE: ovary_diamr **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ovary_diamrv
    .F = "(.F) No Form"
    .M = "(.M) Missing"
    .N = "(.N) Not visualized"
  ;
  ** FORMAT: ovary_vollv **;
  ** FOR VARIABLE: ovary_voll **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ovary_vollv
    .F = "(.F) No Form"
    .M = "(.M) Missing"
    .N = "(.N) Not visualized"
  ;
  ** FORMAT: ovary_volrv **;
  ** FOR VARIABLE: ovary_volr **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ovary_volrv
    .F = "(.F) No Form"
    .M = "(.M) Missing"
    .N = "(.N) Not visualized"
  ;
  ** FORMAT: ovcyst_diamlv **;
  ** FOR VARIABLE: ovcyst_diaml **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ovcyst_diamlv
    .F = "(.F) No Form"
    .M = "(.M) Missing"
    .N = "(.N) Not visualized"
  ;
  ** FORMAT: ovcyst_diamrv **;
  ** FOR VARIABLE: ovcyst_diamr **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ovcyst_diamrv
    .F = "(.F) No Form"
    .M = "(.M) Missing"
    .N = "(.N) Not visualized"
  ;
  ** FORMAT: ovcyst_morphlv **;
  ** FOR VARIABLE: ovcyst_morphl **;
  value ovcyst_morphlv
    0 = "(0) Not visualized"
    1 = "(1) No solid area and outline smooth"
    2 = "(2) Mixed solid area or outline irregular or outline papillaries"
    3 = "(3) Solid area"
  ;
  ** FORMAT: ovcyst_morphrv **;
  ** FOR VARIABLE: ovcyst_morphr **;
  value ovcyst_morphrv
    0 = "(0) Not visualized"
    1 = "(1) No solid area and outline smooth"
    2 = "(2) Mixed solid area or outline irregular or outline papillaries"
    3 = "(3) Solid area"
  ;
  ** FORMAT: ovcyst_outlinelv **;
  ** FOR VARIABLE: ovcyst_outlinel **;
  value ovcyst_outlinelv
    .F = "(.F) No Form"
    .M = "(.M) Missing"
    .N = "(.N) Not visualized"
    1 = "(1) Smooth"
    2 = "(2) Irregularities"
    3 = "(3) Papillarities"
  ;
  ** FORMAT: ovcyst_outlinerv **;
  ** FOR VARIABLE: ovcyst_outliner **;
  value ovcyst_outlinerv
    .F = "(.F) No Form"
    .M = "(.M) Missing"
    .N = "(.N) Not visualized"
    1 = "(1) Smooth"
    2 = "(2) Irregularities"
    3 = "(3) Papillarities"
  ;
  ** FORMAT: ovcyst_solidlv **;
  ** FOR VARIABLE: ovcyst_solidl **;
  value ovcyst_solidlv
    .F = "(.F) No Form"
    .M = "(.M) Missing"
    .N = "(.N) Not visualized"
    0 = "(0) None"
    1 = "(1) Mixed"
    2 = "(2) All Solid"
  ;
  ** FORMAT: ovcyst_solidrv **;
  ** FOR VARIABLE: ovcyst_solidr **;
  value ovcyst_solidrv
    .F = "(.F) No Form"
    .M = "(.M) Missing"
    .N = "(.N) Not visualized"
    0 = "(0) None"
    1 = "(1) Mixed"
    2 = "(2) All Solid"
  ;
  ** FORMAT: ovcyst_sumlv **;
  ** FOR VARIABLE: ovcyst_suml **;
  value ovcyst_sumlv
    0 = "(0) Not visualized"
    1 = "(1) < 3cm, no solid area and smooth outline"
    2 = "(2) < 3cm, mixed solid area or irregular/papillary outline"
    3 = "(3) < 3cm, solid area"
    4 = "(4) 3 -< 6cm, no solid area and smooth outline"
    5 = "(5) 3 -< 6cm, mixed solid area or irregular/papillary outline"
    6 = "(6) 3 -< 6cm, solid area"
    7 = "(7) 6+ cm, no solid area and smooth outline"
    8 = "(8) 6+ cm, mixed solid area or irregular/papillary outline"
    9 = "(9) 6+ cm, solid area"
  ;
  ** FORMAT: ovcyst_sumrv **;
  ** FOR VARIABLE: ovcyst_sumr **;
  value ovcyst_sumrv
    0 = "(0) Not visualized"
    1 = "(1) < 3cm, no solid area and smooth outline"
    2 = "(2) < 3cm, mixed solid area or irregular/papillary outline"
    3 = "(3) < 3cm, solid area"
    4 = "(4) 3 -< 6cm, no solid area and smooth outline"
    5 = "(5) 3 -< 6cm, mixed solid area or irregular/papillary outline"
    6 = "(6) 3 -< 6cm, solid area"
    7 = "(7) 6+ cm, no solid area and smooth outline"
    8 = "(8) 6+ cm, mixed solid area or irregular/papillary outline"
    9 = "(9) 6+ cm, solid area"
  ;
  ** FORMAT: ovcyst_vollv **;
  ** FOR VARIABLE: ovcyst_voll **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ovcyst_vollv
    .F = "(.F) No Form"
    .M = "(.M) Missing"
    .N = "(.N) Not visualized"
  ;
  ** FORMAT: ovcyst_volrv **;
  ** FOR VARIABLE: ovcyst_volr **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value ovcyst_volrv
    .F = "(.F) No Form"
    .M = "(.M) Missing"
    .N = "(.N) Not visualized"
  ;
  ** FORMAT: $examineridv **;
  ** FOR VARIABLE: examinerid **;
  ** CHARACTER VARIABLE (formats not searched/validated) **;
  value $examineridv
  ;
  ** FORMAT: inad_bowv **;
  ** FOR VARIABLE: inad_bow **;
  value inad_bowv
    .M = "(.M) Not Marked"
    1 = "(1) Yes"
  ;
  ** FORMAT: inad_disv **;
  ** FOR VARIABLE: inad_dis **;
  value inad_disv
    .M = "(.M) Not Marked"
    1 = "(1) Yes"
  ;
  ** FORMAT: inad_malv **;
  ** FOR VARIABLE: inad_mal **;
  value inad_malv
    .M = "(.M) Not Marked"
    1 = "(1) Yes"
  ;
  ** FORMAT: inad_othv **;
  ** FOR VARIABLE: inad_oth **;
  value inad_othv
    .M = "(.M) Not Marked"
    1 = "(1) Yes"
  ;
  ** FORMAT: inad_probev **;
  ** FOR VARIABLE: inad_probe **;
  value inad_probev
    .M = "(.M) Not Marked"
    1 = "(1) Yes"
  ;
  ** FORMAT: inad_refv **;
  ** FOR VARIABLE: inad_ref **;
  value inad_refv
    .M = "(.M) Not Marked"
    1 = "(1) Yes"
  ;
  ** FORMAT: medcompv **;
  ** FOR VARIABLE: medcomp **;
  value medcompv
    .F = "(.F) No form"
    .M = "(.M) Missing"
    0 = "(0) No"
    1 = "(1) Yes"
  ;
  ** FORMAT: qamethodv **;
  ** FOR VARIABLE: qamethod **;
  value qamethodv
    .F = "(.F) No form"
    .M = "(.M) Missing"
    1 = "(1) Repeat examination"
    2 = "(2) Observation of examination"
    3 = "(3) Review of files"
  ;
  ** FORMAT: $examinerid_pvis1v **;
  ** FOR VARIABLE: examinerid_pvis1-3 **;
  ** CHARACTER VARIABLE (formats not searched/validated) **;
  value $examinerid_pvis1v
  ;
  ** FORMAT: $examinerid_qvis1v **;
  ** FOR VARIABLE: examinerid_qvis1-3 **;
  ** CHARACTER VARIABLE (formats not searched/validated) **;
  value $examinerid_qvis1v
  ;
  ** FORMAT: inad_bow_p1v **;
  ** FOR VARIABLE: inad_bow_p1-3 **;
  value inad_bow_p1v
    .F = "(.F) No Form"
    .N = "(.N) Not Applicable"
    1 = "(1) Yes"
  ;
  ** FORMAT: inad_bow_q1v **;
  ** FOR VARIABLE: inad_bow_q1-3 **;
  value inad_bow_q1v
    .F = "(.F) No Form"
    .N = "(.N) Not Applicable"
    1 = "(1) Yes"
  ;
  ** FORMAT: inad_dis_p1v **;
  ** FOR VARIABLE: inad_dis_p1-3 **;
  value inad_dis_p1v
    .F = "(.F) No Form"
    .N = "(.N) Not Applicable"
    1 = "(1) Yes"
  ;
  ** FORMAT: inad_dis_q1v **;
  ** FOR VARIABLE: inad_dis_q1-3 **;
  value inad_dis_q1v
    .F = "(.F) No Form"
    .N = "(.N) Not Applicable"
    1 = "(1) Yes"
  ;
  ** FORMAT: inad_mal_p1v **;
  ** FOR VARIABLE: inad_mal_p1-3 **;
  value inad_mal_p1v
    .F = "(.F) No Form"
    .N = "(.N) Not Applicable"
    1 = "(1) Yes"
  ;
  ** FORMAT: inad_mal_q1v **;
  ** FOR VARIABLE: inad_mal_q1-3 **;
  value inad_mal_q1v
    .F = "(.F) No Form"
    .N = "(.N) Not Applicable"
    1 = "(1) Yes"
  ;
  ** FORMAT: inad_oth_p1v **;
  ** FOR VARIABLE: inad_oth_p1-3 **;
  value inad_oth_p1v
    .F = "(.F) No Form"
    .N = "(.N) Not Applicable"
    1 = "(1) Yes"
  ;
  ** FORMAT: inad_oth_q1v **;
  ** FOR VARIABLE: inad_oth_q1-3 **;
  value inad_oth_q1v
    .F = "(.F) No Form"
    .N = "(.N) Not Applicable"
    1 = "(1) Yes"
  ;
  ** FORMAT: inad_probe_p1v **;
  ** FOR VARIABLE: inad_probe_p1-3 **;
  value inad_probe_p1v
    .F = "(.F) No Form"
    .N = "(.N) Not Applicable"
    1 = "(1) Yes"
  ;
  ** FORMAT: inad_probe_q1v **;
  ** FOR VARIABLE: inad_probe_q1-3 **;
  value inad_probe_q1v
    .F = "(.F) No Form"
    .N = "(.N) Not Applicable"
    1 = "(1) Yes"
  ;
  ** FORMAT: inad_ref_p1v **;
  ** FOR VARIABLE: inad_ref_p1-3 **;
  value inad_ref_p1v
    .F = "(.F) No Form"
    .N = "(.N) Not Applicable"
    1 = "(1) Yes"
  ;
  ** FORMAT: inad_ref_q1v **;
  ** FOR VARIABLE: inad_ref_q1-3 **;
  value inad_ref_q1v
    .F = "(.F) No Form"
    .N = "(.N) Not Applicable"
    1 = "(1) Yes"
  ;
  ** FORMAT: phyconsv **;
  ** FOR VARIABLE: phycons **;
  value phyconsv
    .F = "(.F) No form"
    .M = "(.M) Missing"
    0 = "(0) No"
    1 = "(1) Yes"
  ;
  ** FORMAT: physidv **;
  ** FOR VARIABLE: physid **;
  ** CONTINUOUS NUMERIC VARIABLE (formats not searched/validated) **;
  value physidv
    .M = "(.M) Missing"
  ;
run;
