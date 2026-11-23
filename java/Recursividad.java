import java.util.Arrays;

/** Ejercicios de recursividad  VISUALENCE CV. Ejecutar:  java Recursividad.java */
public class Recursividad {

    // 1) Factorial: n! = n * (n-1)!
    static long factorial(int n) { return n <= 1 ? 1 : n * factorial(n - 1); }

    // 2) Fibonacci ingenuo (muestra el costo exponencial) y con memoizacion
    static long fibNaive(int n) { return n < 2 ? n : fibNaive(n - 1) + fibNaive(n - 2); }
    static long fibMemo(int n, long[] memo) {
        if (n < 2) return n;
        if (memo[n] != 0) return memo[n];
        return memo[n] = fibMemo(n - 1, memo) + fibMemo(n - 2, memo);
    }

    // 3) Suma de digitos (p. ej. de un id de publicacion)
    static int sumaDigitos(int n) { return n < 10 ? n : (n % 10) + sumaDigitos(n / 10); }

    // 4) Potencia rapida: x^n en O(log n)
    static double potencia(double x, int n) {
        if (n == 0) return 1;
        double m = potencia(x, n / 2);
        return (n % 2 == 0) ? m * m : m * m * x;
    }

    // 5) Palindromo (ignora mayusculas y espacios)
    static boolean esPalindromo(String s) {
        s = s.toLowerCase().replace(" ", "");
        return s.length() < 2 || (s.charAt(0) == s.charAt(s.length() - 1)
                && esPalindromo(s.substring(1, s.length() - 1)));
    }

    // 6) Busqueda binaria recursiva sobre un arreglo ordenado (p. ej. horas de publicacion)
    static int busquedaBinaria(int[] a, int x, int lo, int hi) {
        if (lo > hi) return -1;
        int mid = (lo + hi) >>> 1;
        if (a[mid] == x) return mid;
        return a[mid] < x ? busquedaBinaria(a, x, mid + 1, hi) : busquedaBinaria(a, x, lo, mid - 1);
    }

    // 7) Maximo y promedio recursivos de un arreglo (p. ej. tasas de interaccion)
    static double maximo(double[] a, int i) { return i == a.length - 1 ? a[i] : Math.max(a[i], maximo(a, i + 1)); }
    static double suma(double[] a, int i) { return i == a.length ? 0 : a[i] + suma(a, i + 1); }

    // 8) Torres de Hanoi: devuelve el numero de movimientos e imprime los pasos
    static int hanoi(int n, char desde, char hacia, char aux, boolean imprimir) {
        if (n == 0) return 0;
        int a = hanoi(n - 1, desde, aux, hacia, imprimir);
        if (imprimir) System.out.println("  mover disco " + n + ": " + desde + " -> " + hacia);
        return a + 1 + hanoi(n - 1, aux, hacia, desde, imprimir);
    }

    public static void main(String[] args) {
        System.out.println("factorial(10) = " + factorial(10));
        System.out.println("fibNaive(25) = " + fibNaive(25) + " | fibMemo(60) = " + fibMemo(60, new long[61]));
        System.out.println("sumaDigitos(78945) = " + sumaDigitos(78945));
        System.out.println("potencia(2, 20) = " + potencia(2, 20));
        System.out.println("esPalindromo(\"Anita lava la tina\") = " + esPalindromo("Anita lava la tina"));
        int[] horas = {7, 9, 12, 13, 18, 19, 20, 21};
        System.out.println("busquedaBinaria(horas, 19) = indice " + busquedaBinaria(horas, 19, 0, horas.length - 1));
        double[] tasas = {0.031, 0.058, 0.044, 0.072, 0.049};
        System.out.printf("maximo = %.3f | promedio = %.4f%n", maximo(tasas, 0), suma(tasas, 0) / tasas.length);
        System.out.println("Hanoi con 3 discos:"); int mov = hanoi(3, 'A', 'C', 'B', true);
        System.out.println("  total de movimientos = " + mov + " (2^3 - 1 = " + ((1 << 3) - 1) + ")");
    }
}
