import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Arbol de decision (CART con impureza de Gini, divisiones multi-rama) construido de forma RECURSIVA.
 * Lee los CSV discretizados que genera 02_dividir.py (java_train.csv, java_val.csv, java_test.csv).
 * Anti-fuga: la profundidad se elige con VALIDACION; TEST se evalua una sola vez.
 * Ejecutar (desde la carpeta java):  java ArbolDecision.java ../data/splits
 */
public class ArbolDecision {

    static final String[] NIVELES = {"bajo", "medio", "alto"};

    static class Nodo {
        int atributo = -1;                       // -1 => hoja
        String mayoria;                          // clase mayoritaria en el nodo
        Map<String, Nodo> hijos = new LinkedHashMap<>();
    }

    static String[] cabecera;

    static List<String[]> leer(Path p) throws IOException {
        List<String> lineas = Files.readAllLines(p);
        cabecera = lineas.get(0).split(",");
        List<String[]> filas = new ArrayList<>();
        for (int i = 1; i < lineas.size(); i++) if (!lineas.get(i).isBlank()) filas.add(lineas.get(i).split(",", -1));
        return filas;
    }

    static String mayoria(List<String[]> d) {
        Map<String, Integer> c = new HashMap<>();
        for (String[] f : d) c.merge(f[f.length - 1], 1, Integer::sum);
        return Collections.max(c.entrySet(), Map.Entry.comparingByValue()).getKey();
    }

    static double gini(List<String[]> d) {
        if (d.isEmpty()) return 0;
        Map<String, Integer> c = new HashMap<>();
        for (String[] f : d) c.merge(f[f.length - 1], 1, Integer::sum);
        double g = 1;
        for (int v : c.values()) { double p = (double) v / d.size(); g -= p * p; }
        return g;
    }

    static Map<String, List<String[]>> dividir(List<String[]> d, int attr) {
        Map<String, List<String[]>> m = new LinkedHashMap<>();
        for (String[] f : d) m.computeIfAbsent(f[attr], k -> new ArrayList<>()).add(f);
        return m;
    }

    /** Construccion recursiva: caso base = nodo puro, profundidad maxima, pocos datos o sin ganancia. */
    static Nodo construir(List<String[]> d, Set<Integer> usados, int prof, int maxProf, int minMuestras) {
        Nodo n = new Nodo(); n.mayoria = mayoria(d);
        double g0 = gini(d);
        if (g0 == 0 || prof >= maxProf || d.size() < minMuestras || usados.size() == cabecera.length - 1) return n;
        int mejor = -1; double mejorG = g0;
        for (int a = 0; a < cabecera.length - 1; a++) {
            if (usados.contains(a)) continue;
            double g = 0;
            for (List<String[]> sub : dividir(d, a).values()) g += (double) sub.size() / d.size() * gini(sub);
            if (g < mejorG - 1e-9) { mejorG = g; mejor = a; }
        }
        if (mejor == -1) return n;                                   // ninguna division reduce la impureza
        n.atributo = mejor;
        Set<Integer> nuevo = new HashSet<>(usados); nuevo.add(mejor);
        for (Map.Entry<String, List<String[]>> e : dividir(d, mejor).entrySet())
            n.hijos.put(e.getKey(), construir(e.getValue(), nuevo, prof + 1, maxProf, minMuestras));  // <- recursion
        return n;
    }

    static String predecir(Nodo n, String[] fila) {
        if (n.atributo == -1) return n.mayoria;
        Nodo h = n.hijos.get(fila[n.atributo]);
        return h == null ? n.mayoria : predecir(h, fila);            // valor no visto => mayoria del nodo
    }

    static void imprimir(Nodo n, String sangria) {
        if (n.atributo == -1) { System.out.println(sangria + "-> " + n.mayoria); return; }
        for (Map.Entry<String, Nodo> e : n.hijos.entrySet()) {
            System.out.println(sangria + cabecera[n.atributo] + " = " + e.getKey());
            imprimir(e.getValue(), sangria + "    ");
        }
    }

    static double exactitud(Nodo arbol, List<String[]> d) {
        int ok = 0; for (String[] f : d) if (predecir(arbol, f).equals(f[f.length - 1])) ok++;
        return (double) ok / d.size();
    }

    static void matriz(Nodo arbol, List<String[]> d) {
        int[][] m = new int[3][3];
        for (String[] f : d) {
            int r = Arrays.asList(NIVELES).indexOf(f[f.length - 1]), p = Arrays.asList(NIVELES).indexOf(predecir(arbol, f));
            if (r >= 0 && p >= 0) m[r][p]++;
        }
        System.out.println("Matriz de confusion (filas = real, columnas = predicho):");
        System.out.printf("%10s %7s %7s %7s%n", "", NIVELES[0], NIVELES[1], NIVELES[2]);
        double f1 = 0;
        for (int i = 0; i < 3; i++) {
            System.out.printf("%10s %7d %7d %7d%n", NIVELES[i], m[i][0], m[i][1], m[i][2]);
            int tp = m[i][i], fp = 0, fn = 0;
            for (int j = 0; j < 3; j++) { if (j != i) { fp += m[j][i]; fn += m[i][j]; } }
            double pr = tp + fp == 0 ? 0 : (double) tp / (tp + fp), rc = tp + fn == 0 ? 0 : (double) tp / (tp + fn);
            f1 += pr + rc == 0 ? 0 : 2 * pr * rc / (pr + rc);
        }
        System.out.printf("F1 macro = %.3f%n", f1 / 3);
    }

    public static void main(String[] args) throws IOException {
        Path dir = Paths.get(args.length > 0 ? args[0] : "../data/splits");
        List<String[]> train = leer(dir.resolve("java_train.csv"));
        List<String[]> val = leer(dir.resolve("java_val.csv"));
        List<String[]> test = leer(dir.resolve("java_test.csv"));
        System.out.printf("train=%d val=%d test=%d%n", train.size(), val.size(), test.size());

        int mejorProf = 1; double mejorVal = -1;                      // seleccion de profundidad con VALIDACION
        for (int p = 1; p <= 8; p++) {
            Nodo a = construir(train, new HashSet<>(), 0, p, 10);
            double acc = exactitud(a, val);
            System.out.printf("profundidad %d -> exactitud val = %.3f%n", p, acc);
            if (acc > mejorVal) { mejorVal = acc; mejorProf = p; }
        }
        Nodo arbol = construir(train, new HashSet<>(), 0, mejorProf, 10);
        System.out.println("\nProfundidad elegida (segun validacion): " + mejorProf);
        System.out.println("\n=== Arbol ===");
        imprimir(arbol, "");
        System.out.printf("%n=== TEST (una sola evaluacion) ===%nExactitud = %.3f%n", exactitud(arbol, test));
        matriz(arbol, test);
    }
}
