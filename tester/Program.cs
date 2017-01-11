﻿using math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace tester
{
    class Program
    {

        double[,] X = new double[,] {
        {0.2, 1.020562258},
        {0.43, 1.218506908},
        {0.32, 0.184617471},
    {0.53, 1.534720884},
    {0.59, 0.822312634},
    {0.678, 0.419922912},
    {0.766, -0.216603169},
    {0.854, -0.583984585},
    {0.942, 1.698239806},
    {1.03, 0.953863041},
    {1.118, 1.875333287},
    {1.206, 1.132548225},
    {1.294, 2.058306355},
    {1.382, 0.293087256},
    {1.47, 0.444208613},
    {1.558, 1.892133761},
    {1.646, 0.99},
    {1.734, 0.93},
    {1.822, 0.988834473},
    {1.91, 0.948740604},
    {1.998, 0.908646735},
    {2.086, 0.868552867},
    {2.174, 0.828458998}};




        static void Main(string[] args)
        {
            /*
            Matrix a = new Matrix(3, 1);
            a.FillRandom(-1.0, 1.0);
            Matrix b = new Matrix(3);
            b.FillRandom(1.0, 2.0);
            a.Print();
            b.Print();  
            Matrix p = a.Dot(b);
            p.Print();
            p = b.Dot(a);
            p.Print();

            Matrix c = new Matrix(2, 3);
            c.Map((x) => x + 1);
            c.Print();
            c.Map((x) => Math.Tanh(x));
            c.Print();
            Matrix.Transpose(c).Print();
            */
            Matrix x = new Matrix(3, 2);
            x.FillRandom(0.1, 1.1);
            x.Print("X=");
            //x.Row(1).Print("row(1)");
            //x.Column(1).Print("column(1)");

            Matrix y = new Matrix(3, 1);
            y.FillRandom(0.1, 1.0);
            y.Print("Y=");
            Matrix w0 = new Matrix(2, 2);
            w0.FillRandom(-0.5, 0.5);
            Matrix w1 = new Matrix(2, 1);
            w1.FillRandom(-0.5, 0.5);
            //
            Matrix a1 = null;
            for (int i = 0; i < x.Rows; i++)
            {
                w0.Print("w0=");
                Matrix a0 = x.Row(i).Dot(w0);
                a0.Map((v) => Math.Tanh(v));
                a0.Print("a0=");
                w1.Print("w1=");
                a1 = a0.Dot(w1);
                a1.Map((v) => Math.Tanh(v));
                a1.Print("a1=");
                Matrix e = y.Row(i) - a1;
                Console.WriteLine();
                Console.WriteLine("=====> Error = {0}", e.SquaredError());
                // BP
                Matrix d2 = e * a1.MapNew(Prime);
                d2.Print("d2=");
                Matrix d1 = d2.Dot(w1.Transpose()) * a0.MapNew(Prime);
                d1.Print("d1=");
                Matrix d0 = d1.Dot(w0.Transpose()) * x.Row(i).MapNew(Prime);
                d0.Print("d0=");
                // update weights
                w0 = w0 + d0.Transpose().Dot(a0);
                w0.Print("w0=");
                w1 = w1 + d1.Transpose().Dot(a1);
                w1.Print("w1=");
            }

        }

        public static double Prime(double x)
        {
            return (1.0 - Math.Pow(Math.Tanh(x), 2.0));
        }
    }
}
