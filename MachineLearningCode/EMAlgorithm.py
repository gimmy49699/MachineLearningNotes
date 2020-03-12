'''
    EM-Algorithm is a method to infer parameters with hidden varible. Applications like K-means and GMM, et, al.
    EM-Algorithm:
        Independent variable: x; Hidden variable: z; parameter: θ, Tolerance: ε.
        Maximum likelihood function: L(θ) = ∑logP(x|θ); P(x|θ) = P(x,z|θ)
            L(θ) = ∑logP(x|θ)
                 = ∑logP(x,z|θ)
                 = ∑log∑P(z|θ)P(x|z,θ)
            L(θ) - L(θ`) = ∑log∑P(z|θ)P(x|z,θ) - ∑logP(x|θ`)                            suppose L(θ`) is known
                         = ∑log∑P(z|x,θ`)*{(P(z|θ)P(x|z,θ))/P(z|x,θ`)} - ∑logP(x|θ`)
                         ≥ ∑∑P(z|x,θ`)*log{(P(z|θ)P(x|z,θ))/P(z|x,θ`)} - ∑logP(x|θ`)    jensen inequation
                         = ∑∑P(z|x,θ`)*log{(P(z|θ)P(x|z,θ))/P(z|x,θ`)P(x|θ`)}           ∑logP(x|θ`) = ∑∑P(z|x,θ`)logP(x|θ`), ∑P(z|x,θ`) = 1
            ∴ L(θ) ≥ L(θ`) + Q(θ|θ`).
              where Q(θ|θ`) = ∑∑P(z|x,θ`)*log{(P(z|θ)P(x|z,θ))/P(z|x,θ`)P(x|θ`)}.
        EM:
            1). initialize parameter θ.
            2). E-step: calculate P(z|x,θ)
            3). M-step: maximize Q(θ|θ`) based on P(z|x,θ) calculated in 2).
            4). stop if |θ - θ`| ≤ ε, else go to 2).

    GMM(Gaussian Mixture Model):
'''