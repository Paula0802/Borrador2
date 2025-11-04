from math import cos, pi, radians, sin
import cmath  # use stdlib cmath for complex math (compatible with numpy arrays)

from numpy import array, identity, matmul


# Modulo para obtener la matriz de transferencia


class TransferMatrixMethod(object):
    """
    This class implements the transfer matrix method
    """

    def __init__(self, theta, l, n, thicknesses, polarization):
        self.theta: float = theta  # angulo
        self.l: float = l  # longitud de onda
        self.n: list = n  # indices de refracciòn
        self.thicknesses: list = thicknesses  # espesores
        self.polarization: str = polarization  # tipo de polarizaciòn 

    def get_propagation_vectors(self):
        """
        This method:
        *Calculates the initial propagation vector k_0 and
        the propagation vector in the z-component of the system for each
        angle and one  lamda, both of which are held constant
        *Receives the angle (theta (float)), wave length (float), refractive index
        of the first medium  (n0 (complex or float))
        *Returns k0 and kz (complex or floats )
        """
        if self.l <= 0:
            raise ValueError(f"{self.l} l is not valid")

        i = (2 * pi) / self.l
        setattr(self, "propagation_vector_i", i)
        theta_radians = radians(self.theta)
        z = (self.n[0] * i) * sin(theta_radians)
        setattr(self, "propagation_vector_z", z)
        return i, z

    def get_propagation_vectors_x(self):
        """
        This method:
        *Calculates the propagations vectors of the system in x- component
        *Recive the list  list_ni of the refractive index (complex or floats),
        k0 and kz (calculated in the previous function (complex or floats))
        *Returns  a list of the propations vectors in the x-component (complex),
        returns an element by per material
        """
        list_ = []
        self.get_propagation_vectors()  # llamada recursiva

        for i in range(0, len(self.n)):
            a = (self.n[i] * self.propagation_vector_i) ** 2
            b = (self.propagation_vector_z) ** 2
            c = a - b
            kx = cmath.sqrt(c)
            list_.append(complex(kx))
        setattr(self, "propagation_vectors_x", list_)
        return list_

    def get_phi(self):
        """
        This method:
        *Calculates the phi factor
        *Recive  the list  list_ni of the refractive index (complexex and/or floats),
        a list with the thicknesses of all system layers
        *Returns a list phi (complexes), returns an element by per layer
        """
        list_ = []
        self.get_propagation_vectors()

        if len(self.thicknesses) > 0:
            theta_rad = radians(self.theta)
            for i in range(0, len(self.thicknesses)):
                a = self.propagation_vector_i * self.n[i + 1] * self.thicknesses[i]
                b = self.n[0] / self.n[i + 1]
                s = sin(theta_rad)
                d = a * cmath.sqrt(1 - (b * s) ** 2)
                list_.append(complex(d))
        setattr(self, "list_phi", list_)
        return list_

    def get_reflection_fresnel_coefficients(self):
        """
        This method:
        *Calculates the reflection fresnel coefficients (rij)
        *Recive type of polarization (str), a list_kx (calculated in the one
        previous function (complex or floats)),
        a list_ni (complexes)
        *Returns the list_rij (complex), returns an element by per interface
        """
        list_ = []
        self.get_propagation_vectors_x()

        if self.polarization == "S":
            for i in range(0, len(self.n) - 1):
                a = self.propagation_vectors_x[i] - self.propagation_vectors_x[i + 1]
                b = self.propagation_vectors_x[i] + self.propagation_vectors_x[i + 1]
                r = a / b
                list_.append(r)
        else:
            for i in range(0, len(self.n) - 1):
                a = (self.n[i] ** 2) * self.propagation_vectors_x[i + 1]
                b = (self.n[i + 1] ** 2) * self.propagation_vectors_x[i]
                r = (a - b) / (a + b)
                list_.append(r)
        setattr(self, "reflection_coefficients", list_)
        return list_

    def get_trasmission_fresnel_coefficients(self):
        """
        This method:
        *Calculates the transmission fresnel coefficient (tij)
        *Recive type of polarization (str) and the lis of reflection fresnel
        coefficients (calculated in the previous function (complex or floats))
        *Returns the list_tij (complex), returns an element by per interface
        """
        list_ = []
        self.get_reflection_fresnel_coefficients()

        if self.polarization == "S":
            for i in range(0, len(self.n) - 1):
                tij = 1 + self.reflection_coefficients[i]
                list_.append(tij)
        else:
            for i in range(0, len(self.n) - 1):
                a = self.n[i] / self.n[i + 1]
                b = 1 + self.reflection_coefficients[i]
                tij = a * b
                list_.append(tij)
        setattr(self, "trasmission_coefficients", list_)
        return list_

    def get_dinamical_matriz(self):

        """
        This method:
        *Calculates the dinamical matriz (complexes)
        *Recive two list the list_rij and list_tij (calculated in the previous functions )
        *Retur one list of matrix (complex or float) returns an element by per element in the list_rij
        """
        list_ = []
        self.get_reflection_fresnel_coefficients()
        self.get_trasmission_fresnel_coefficients()

        for i, j in zip(self.reflection_coefficients, self.trasmission_coefficients):
            a = array([[1 / j, i / j], [i / j, 1 / j]])
            list_.append(a)
        setattr(self, "dinamical_matrices", list_)
        return list_

    def get_propagation_matriz(self):

        """
        This method:
        *Calculates the propagation matriz (complexes)
        *Recive list phi (complexes)
        *Retur the a list of matriz (complexes) one for each layer of the system
        """
        list_ = []
        self.get_phi()

        if len(self.thicknesses) > 0:
            for i in self.list_phi:
                a = array([[cmath.exp(-1j * i), 0], [0, cmath.exp(1j * i)]])
                list_.append(a)
        setattr(self, "propagation_matriz", list_)
        return list_

    def get_transfer_matriz(self):

        """
        This Funtion:
        *calculates the multiplication between the dynamic matrices and the propagation matrices
        *Recive two list list_pm and list_dm
        *Returns a 2*2 matrix with complex elements

        """
        self.get_dinamical_matriz()
        self.get_propagation_matriz()
        m = identity(2)

        if not (len(self.propagation_matriz) == 0):
            for i in range(0, len(self.propagation_matriz)):
                m = matmul(m, self.dinamical_matrices[i])
                m = matmul(m, self.propagation_matriz[i])
            m = matmul(m, self.dinamical_matrices[-1])
        else:
            for i in range(0, len(self.dinamical_matrices)):
                m = matmul(m, self.dinamical_matrices[i])
        setattr(self, "transfer_matriz", m)
        return m
        
    def get_reflection_amplitude(self):
        """
        Devuelve la amplitud compleja de reflexión r (no |r|^2).
        """
        self.get_transfer_matriz()
        # r = m[1,0] / m[0,0]  (como en get_reflectance)
        r = (self.transfer_matriz[1][0]) / (self.transfer_matriz[0][0])
        setattr(self, "r_complex", complex(r))
        return complex(r)

    def get_transmission_amplitude(self):
        """
        Devuelve la amplitud compleja de transmisión t (compleja).
        Si quieres la transmitancia energética usa get_transmittance() (ya existe).
        """
        self.get_transfer_matriz()
        t = 1.0 / self.transfer_matriz[0][0]
        setattr(self, "t_complex", complex(t))
        return complex(t)


    def get_reflectance(self):
        """
        This method:
        *calculates the reflectance of the multilayer system.
        *Recive one matriz (transfer matriz)
        *Returns a element real and posotive
        """
        
        self.get_transfer_matriz()

        r = (self.transfer_matriz[1][0]) / (self.transfer_matriz[0][0])
        reflectance = (abs(r)) ** 2
        setattr(self, "reflectance", reflectance)
        return float(reflectance)

    def get_transmittance(self):
        """
        This method:
        *calculates the trasmitance of the multilayer system.
        *Recive one matriz (transfer matriz) and the list  n of the refractive index (complex or floats)
        and the list of cosine of the angles at which the wave reaches each layer (complex or floats)
        *Returns a element complex or float
        """
        
        theta_radians = radians(self.theta)
        self.get_transfer_matriz()  
        a = self.n[0] / self.n[-1]
        b = cmath.sqrt(1 - (a * sin(theta_radians)) ** 2)
        
        c = self.n[-1] * b
        d = self.n[0] * cos(theta_radians)
        e = c / d
        f = 1 / self.transfer_matriz[0][0]
        f = (abs(f)) ** 2
        
        g =  e*f
        t = g.real 

        setattr(self, "trasmittance", t)
        return float(t)
 
 
        

    def get_absortance(self):
        """
        This method:
        *calculates the absortance of the multilayer system.
        *Recive  reflectance (float) and transmittance (float or complex )
        *Returns a element complex or float"""
        self.get_reflectance()
        self.get_transmittance()

        absortance = 1 - self.reflectance - self.trasmittance
        absortance = abs(absortance)
        return float(absortance)

