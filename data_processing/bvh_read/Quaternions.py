""" This code was taken from Daniel Holden
 at his web-site http://theorangeduck.com  """

import numpy as np

class Quaternions:
    """
    Quaternions is a wrapper around a numpy ndarray
    that allows it to act as if it were an narray of
    a quaternion data type.
    
    Therefore addition, subtraction, multiplication,
    division, negation, absolute, are all defined
    in terms of quaternion operations such as quaternion
    multiplication.
    
    This allows for much neater code and many routines
    which conceptually do the same thing to be written
    in the same way for point data and for rotation data.
    
    The Quaternions class has been desgined such that it
    should support broadcasting and slicing in all of the
    usual ways.
    """
    
    def __init__(self, qs):
        if isinstance(qs, np.ndarray):
        
            if len(qs.shape) == 1: qs = np.array([qs])
            self.qs = qs
            return
            
        if isinstance(qs, Quaternions):
            self.qs = qs.qs
            return
            
        raise TypeError('Quaternions must be constructed from iterable, numpy array, or Quaternions, not %s' % type(qs))
    
    def __str__(self): return "Quaternions("+ str(self.qs) + ")"
    def __repr__(self): return "Quaternions("+ repr(self.qs) + ")"
    
    """ Helper Methods for Broadcasting and Data extraction """

    @classmethod
    def _broadcast(cls, sqs, oqs, scalar=False):

        if isinstance(oqs, float): return sqs, oqs * np.ones(sqs.shape[:-1])

        ss = np.array(sqs.shape) if not scalar else np.array(sqs.shape[:-1])
        os = np.array(oqs.shape)

        if len(ss) != len(os):
            raise TypeError('Quaternions cannot broadcast together shapes %s and %s' % (sqs.shape, oqs.shape))

        if np.all(ss == os): return sqs, oqs

        if not np.all((ss == os) | (os == np.ones(len(os))) | (ss == np.ones(len(ss)))):
            raise TypeError('Quaternions cannot broadcast together shapes %s and %s' % (sqs.shape, oqs.shape))

        sqsn, oqsn = sqs.copy(), oqs.copy()

        for a in np.where(ss == 1)[0]: sqsn = sqsn.repeat(os[a], axis=a)
        for a in np.where(os == 1)[0]: oqsn = oqsn.repeat(ss[a], axis=a)

        return sqsn, oqsn

    def __getitem__(self, k):    return Quaternions(self.qs[k]) 
    def __setitem__(self, k, v): self.qs[k] = v.qs

    def __iter__(self): return iter(self.qs)
    def __len__(self): return len(self.qs)
        
    @property
    def lengths(self):
        return np.sum(self.qs**2.0, axis=-1)**0.5
    
    @property
    def reals(self):
        return self.qs[...,0]
        
    @property
    def imaginaries(self):
        return self.qs[...,1:4]
    
    @property
    def shape(self): return self.qs.shape[:-1]

    def dot(self, q): return np.sum(self.qs * q.qs, axis=-1)
    
    def copy(self): return Quaternions(np.copy(self.qs))

    def normalized(self):
        return Quaternions(self.qs / self.lengths[...,np.newaxis])

    """ Quaterion Multiplication """

    def __mul__(self, other):
        """
        Quaternion multiplication has three main methods.

        When multiplying a Quaternions array by Quaternions
        normal quaternion multiplication is performed.

        When multiplying a Quaternions array by a vector
        array of the same shape, where the last axis is 3,
        it is assumed to be a Quaternion by 3D-Vector
        multiplication and the 3D-Vectors are rotated
        in space by the Quaternions.

        When multipplying a Quaternions array by a scalar
        or vector of different shape it is assumed to be
        a Quaternions by Scalars multiplication and the
        Quaternions are scaled using Slerp and the identity
        quaternions.
        """

        """ If Quaternions type do Quaternions * Quaternions """
        if isinstance(other, Quaternions):
            sqs, oqs = Quaternions._broadcast(self.qs, other.qs)

            q0 = sqs[..., 0];
            q1 = sqs[..., 1];
            q2 = sqs[..., 2];
            q3 = sqs[..., 3];
            r0 = oqs[..., 0];
            r1 = oqs[..., 1];
            r2 = oqs[..., 2];
            r3 = oqs[..., 3];

            qs = np.empty(sqs.shape)
            qs[..., 0] = r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3
            qs[..., 1] = r0 * q1 + r1 * q0 - r2 * q3 + r3 * q2
            qs[..., 2] = r0 * q2 + r1 * q3 + r2 * q0 - r3 * q1
            qs[..., 3] = r0 * q3 - r1 * q2 + r2 * q1 + r3 * q0

            return Quaternions(qs)

        """ If array type do Quaternions * Vectors """
        if isinstance(other, np.ndarray) and other.shape[-1] == 3:
            vs = Quaternions(np.concatenate([np.zeros(other.shape[:-1] + (1,)), other], axis=-1))
            return (self * (vs * -self)).imaginaries

        """ If float do Quaternions * Scalars """
        if isinstance(other, np.ndarray) or isinstance(other, float):
            return Quaternions.slerp(Quaternions.id_like(self), self, other)

        raise TypeError('Cannot multiply/add Quaternions with type %s' % str(type(other)))

    
    def euler(self, order='xyz'):
        
        q = self.normalized().qs
        q0 = q[...,0]
        q1 = q[...,1]
        q2 = q[...,2]
        q3 = q[...,3]
        es = np.zeros(self.shape + (3,))
        
        if   order == 'xyz':
            es[...,0] = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
            es[...,1] = np.arcsin((2 * (q0 * q2 - q3 * q1)).clip(-1,1))
            es[...,2] = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
        elif order == 'yzx':
            es[...,0] = np.arctan2(2 * (q1 * q0 - q2 * q3), -q1 * q1 + q2 * q2 - q3 * q3 + q0 * q0)
            es[...,1] = np.arctan2(2 * (q2 * q0 - q1 * q3),  q1 * q1 - q2 * q2 - q3 * q3 + q0 * q0)
            es[...,2] = np.arcsin((2 * (q1 * q2 + q3 * q0)).clip(-1,1))
        else:
            raise NotImplementedError('Cannot convert from ordering %s' % order)
        
        return es

    def ravel(self):
        return self.qs.ravel()
    
    @classmethod
    def id(cls, n):
        
        if isinstance(n, tuple):
            qs = np.zeros(n + (4,))
            qs[...,0] = 1.0
            return Quaternions(qs)
        
        if isinstance(n, int) or isinstance(n, long):
            qs = np.zeros((n,4))
            qs[:,0] = 1.0
            return Quaternions(qs)
        
        raise TypeError('Cannot Construct Quaternion from %s type' % str(type(n)))

        
    @classmethod
    def exp(cls, ws):
    
        ts = np.sum(ws**2.0, axis=-1)**0.5
        ts[ts == 0] = 0.001
        ls = np.sin(ts) / ts
        
        qs = np.empty(ws.shape[:-1] + (4,))
        qs[...,0] = np.cos(ts)
        qs[...,1] = ws[...,0] * ls
        qs[...,2] = ws[...,1] * ls
        qs[...,3] = ws[...,2] * ls
        
        return Quaternions(qs).normalized()
    
    @classmethod
    def from_euler(cls, es, order='xyz', world=False):
    
        axis = {
            'x' : np.array([1,0,0]),
            'y' : np.array([0,1,0]),
            'z' : np.array([0,0,1]),
        }
        
        q0s = Quaternions.from_angle_axis(es[...,0], axis[order[0]])
        q1s = Quaternions.from_angle_axis(es[...,1], axis[order[1]])
        q2s = Quaternions.from_angle_axis(es[...,2], axis[order[2]])
        
        return (q2s * (q1s * q0s)) if world else (q0s * (q1s * q2s))

    @classmethod
    def from_angle_axis(cls, angles, axis):
        axis = axis / (np.sqrt(np.sum(axis ** 2, axis=-1)) + 1e-10)[..., np.newaxis]
        sines = np.sin(angles / 2.0)[..., np.newaxis]
        cosines = np.cos(angles / 2.0)[..., np.newaxis]
        return Quaternions(np.concatenate([cosines, axis * sines], axis=-1))
    
    