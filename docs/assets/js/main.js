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

// L'apparition des cartes est desormais entierement geree en CSS
// (@keyframes card-appear, animation-fill-mode: both) : le contenu reste
// visible meme si le JavaScript ne s'execute pas ou est interrompu.
