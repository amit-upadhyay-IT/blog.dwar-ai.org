// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightBlog from 'starlight-blog';
import starlightThemeVintage from 'starlight-theme-vintage';

// https://astro.build/config
export default defineConfig({
	site: 'https://blog.dwar-ai.org', // Your actual domain
	integrations: [
		starlight({
			title: 'My Technical Blog',
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/withastro/starlight' }],
			plugins: [
				starlightThemeVintage(),
				starlightBlog(),
			],
			head: [
				{
					tag: 'script',
					attrs: { type: 'module' },
					content: `
						import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
						mermaid.initialize({ startOnLoad: false, theme: 'dark' });

						document.addEventListener('DOMContentLoaded', async () => {
							const mermaidBlocks = document.querySelectorAll('pre[data-language="mermaid"]');
							mermaidBlocks.forEach((pre) => {
								const wrapper = pre.closest('div.expressive-code');
								
								// Expressive code splits lines into .ec-line divs. We extract them to preserve newlines and avoid HTML entity encoding issues in data-code.
								const lines = pre.querySelectorAll('.ec-line');
								let content = '';
								if (lines.length > 0) {
									content = Array.from(lines).map(line => line.textContent).join(String.fromCharCode(10));
								} else {
									content = pre.textContent;
								}
								
								const div = document.createElement('div');
								div.className = 'mermaid';
								div.textContent = content;
								
								if (wrapper) {
									wrapper.parentNode.replaceChild(div, wrapper);
								} else {
									pre.parentNode.replaceChild(div, pre);
								}
							});
							await mermaid.run({ querySelector: '.mermaid' });
						});
					`,
				},
			],
			sidebar: [
				{
					label: 'Guides',
					items: [
						{ label: 'Example Guide', slug: 'guides/example' },
					],
				},
				{
					label: 'Reference',
					items: [{ autogenerate: { directory: 'reference' } }],
				},
			],
		}),
	],
});
