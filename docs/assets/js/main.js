// Défilement doux vers les ancres (l'offset de la barre collante est géré en CSS
// via scroll-margin-top, ce qui évite de recalculer une hauteur en JS).
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        const href = this.getAttribute('href');
        if (href === '#') {
            return;
        }
        const target = document.querySelector(href);
        if (target) {
            e.preventDefault();
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            history.replaceState(null, '', href);
        }
    });
});

// Surligne l'entrée de navigation correspondant à la section visible.
const navLinks = Array.from(document.querySelectorAll('.nav-links a[href^="#"]'));
const sections = navLinks
    .map(link => document.querySelector(link.getAttribute('href')))
    .filter(Boolean);

if (sections.length) {
    const navObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (!entry.isIntersecting) {
                return;
            }
            navLinks.forEach(link => {
                link.classList.toggle(
                    'is-active',
                    link.getAttribute('href') === '#' + entry.target.id
                );
            });
        });
    }, { rootMargin: '-40% 0px -55% 0px', threshold: 0 });

    sections.forEach(section => navObserver.observe(section));
}

// Anime les barres de métriques à l'entrée dans le viewport.
const metricObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (!entry.isIntersecting) {
            return;
        }
        const fill = entry.target.querySelector('.metric-fill');
        if (fill) {
            const width = fill.style.width;
            fill.style.width = '0%';
            setTimeout(() => { fill.style.width = width; }, 100);
        }
        metricObserver.unobserve(entry.target);
    });
}, { threshold: 0.4 });

document.querySelectorAll('.metric-card').forEach(card => metricObserver.observe(card));

// Apparition progressive des cartes.
// Le seuil est volontairement bas : une carte plus haute que le viewport
// (mobile) doit malgré tout devenir visible.
const fadeObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (!entry.isIntersecting) {
            return;
        }
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
        fadeObserver.unobserve(entry.target);
    });
}, { threshold: 0.1 });

document.querySelectorAll('.solution-card, .impact-card, .stat-card, .viz-card').forEach(card => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(20px)';
    card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    fadeObserver.observe(card);
});

// Filet de sécurité : si l'IntersectionObserver n'est pas disponible ou si un
// élément reste masqué, on révèle tout après le chargement complet.
window.addEventListener('load', () => {
    setTimeout(() => {
        document.querySelectorAll('.solution-card, .impact-card, .stat-card, .viz-card')
            .forEach(card => {
                if (card.style.opacity === '0') {
                    card.style.opacity = '1';
                    card.style.transform = 'translateY(0)';
                }
            });
    }, 1500);
});
