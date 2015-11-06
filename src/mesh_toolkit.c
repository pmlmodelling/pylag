/* 
 * Compute barycentric coordinates
 */
void get_barycentric_coords(double x, double y, double* x_nodes, 
        double* y_nodes, double* phi)
{
    // Array entries
    double a11 = y_nodes[2] - y_nodes[0];
    double a12 = x_nodes[0] - x_nodes[2];
    double a21 = y_nodes[0] - y_nodes[1];
    double a22 = x_nodes[1] - x_nodes[0];
    
    // Determinant
    double det = a11 * a22 - a12 * a21;
    
    // Transformation to barycentric coordinates
    phi[0] = (a11*(x - x_nodes[0]) + a12*(y - y_nodes[0]))/det;
    phi[1] = (a21*(x - x_nodes[0]) + a22*(y - y_nodes[0]))/det;
    phi[2] = 1.0 - phi[0] - phi[1];
}