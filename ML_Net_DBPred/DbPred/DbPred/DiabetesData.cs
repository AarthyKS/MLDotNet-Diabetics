using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Runtime.Api;

namespace DbPred
{
    public class DiabetesData
    {
        [Column(ordinal: "0", name: "num_preg")]
        public float num_preg;
        [Column(ordinal: "1", name: "glucose_conc")]
        public float glucose_conc;
        [Column(ordinal: "2", name: "diastolic_bp")]
        public float diastolic_bp;
        [Column(ordinal: "3", name: "thickness")]
        public float thickness;
        [Column(ordinal: "4", name: "insulin")]
        public float insulin;
        [Column(ordinal: "5", name: "bmi")]
        public float bmi;
        [Column(ordinal: "6", name: "diab_pred")]
        public float diab_pred;
        [Column(ordinal: "7", name: "age")]
        public float age;
        [Column(ordinal: "8", name: "Label")]
        public bool diabetes;
    }
    
    public class DiabetesPrediction
    {
        [ColumnName("Label")]
        public bool diabetes;
    }
}
